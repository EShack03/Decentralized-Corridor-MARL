"""
[ES] added (17 Nov 2026) - distillation, data collection (one DAgger round, one layout / count).

Flies episodes and records, for every second an aircraft is still flying, what the TEACHER would do:
the ORIGINAL policy with the final safety filter's mask applied (perfect information), i.e. the move the
filtered system would actually fly. The aircraft are flown by a mixture (per aircraft, per second) of the
teacher (probability beta) and the STUDENT network acting alone without a filter (probability 1 - beta),
as in DAgger (Ross, Gordon & Bagnell 2011). Round 0 uses beta = 1 (teacher flies).

Nothing original is changed: the env, the runner, the policy and the filter are built exactly as
eval_mpe.py builds them (same functions), and the filter mask is applied the same way as the runner's
single-policy render path (copied into _filter_mask below; the runner file is untouched). The original
weights are read from --model_dir and never written.

Writes one compressed .npz shard (--distill_out) with, per kept aircraft-second:
  obs, node_obs, agent_id, rnn (the STUDENT's recurrent state at that second), label (teacher move),
  orig_logits (original policy's unmasked logits), fmask (moves the filter allows), interv (filter changed
  the original policy's move), blocked (filter removed moves but not the policy's choice), w (1 / keep
  probability), adj_idx (index into the per-second adjacency array).
All interventions are kept; the other seconds are subsampled (--distill_keep_blocked, --distill_keep_frac) to
keep shards small, and w restores their true share in the loss.

Usage (repo root, venv active) - the usual eval arguments plus the final-filter flags, e.g.
  python onpolicy/scripts/distill_collect.py <eval args> <final filter args> \
      --distill_out data/r0_merge_N20_s1001.npz --distill_beta 1.0 [--distill_student <weights dir>]
"""
import argparse
import copy
import os
import sys
import time

sys.path.append(os.path.abspath(os.getcwd()))

import numpy as np
import torch

from onpolicy.config import get_config
from onpolicy.scripts.eval_mpe import make_render_env, parse_args, modify_args, _attach_cbf_eval

SEP_KM = 0.1524      # separation standard (500 ft), for the loss-of-separation monitor only


def distill_args(argv):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--distill_out", type=str, required=True)
    p.add_argument("--distill_student", type=str, default="",
                   help="folder with the student's actor.pt (empty = student is the original policy)")
    p.add_argument("--distill_beta", type=float, default=1.0, help="probability the teacher flies a given aircraft-second")
    p.add_argument("--distill_keep_frac", type=float, default=0.1,
                   help="fraction kept of aircraft-seconds where the filter removed nothing (all interventions are kept)")
    p.add_argument("--distill_keep_blocked", type=float, default=0.5,
                   help="fraction kept of aircraft-seconds where the filter removed moves but not the policy's choice")
    return p.parse_known_args(argv)[0]


# ---------------------------------------------------------------- network helpers
def actor_logits(actor, obs, node_obs, adj, agent_id, rnn, masks):
    """Unmasked action logits and the new recurrent state, computed exactly as GR_Actor.forward does
    (GNN -> concat -> MLP -> GRU -> final linear layer of the categorical head)."""
    tp = actor.tpdv
    obs = torch.as_tensor(obs).to(**tp); node_obs = torch.as_tensor(node_obs).to(**tp)
    adj = torch.as_tensor(adj).to(**tp); agent_id = torch.as_tensor(agent_id).to(**tp).long()
    rnn = torch.as_tensor(rnn).to(**tp); masks = torch.as_tensor(masks).to(**tp)
    feats = actor.base(torch.cat([obs, actor.gnn_base(node_obs, adj, agent_id)], dim=1))
    if actor._use_naive_recurrent_policy or actor._use_recurrent_policy:
        feats, rnn = actor.rnn(feats, rnn, masks)
    return actor.act.action_out.linear(feats), rnn


def masked_argmax(logits, avail):
    """The deterministic action of the policy with available_actions (= argmax over the allowed moves)."""
    x = logits.clone()
    x[torch.as_tensor(avail, device=x.device) == 0] = torch.finfo(x.dtype).min
    return torch.softmax(x, -1).argmax(-1)


# ---------------------------------------------------------------- filter mask (copy of the runner's block)
def _filter_mask(runner, envs, node_obs, adj, available_actions):
    """Same computation as the single-policy render path in graph_mpe_runner.py (CBF eval filter block)."""
    from onpolicy.algorithms.utils import cbf_eval_metrics as _cbfm
    from onpolicy.algorithms.utils.cbf_filter import route_paths, route_paths_map
    world = envs.envs[0].world
    wa = world.agents
    wstate = np.array([[a.state.p_pos[0], a.state.p_pos[1], float(a.state.theta), float(a.state.speed)]
                       for a in wa], dtype=np.float32)[None]
    scen = runner.cbf_eval_scen
    world_cbf, wtrue = world, None
    sensing = getattr(runner, "cbf_sensing", None)
    if sensing is not None:
        from onpolicy.algorithms.utils.cbf_sensing import SensedWorld
        wtrue = wstate
        wstate = sensing.apply(wtrue[0], parked=np.array([bool(getattr(a, "status", False)) for a in wa]))[None]
        world_cbf = SensedWorld(world, wstate[0])
    win_tube = None
    try:
        inc = _cbfm.in_corridor_flags(scen, world_cbf, wa)
        if inc is not None:
            win_tube = np.asarray(inc, dtype=bool)[None]
    except Exception:
        win_tube = None
    routes = None
    nm = getattr(runner.cbf_eval_filter, "neighbour_model", "")
    if nm == "route":
        routes = [route_paths(scen, world_cbf)]
    elif nm == "route_map":
        routes = [route_paths_map(scen, world_cbf)]
    mask, no_safe = runner.cbf_eval_filter.masks_for_threads(
        np.concatenate(node_obs)[None], np.concatenate(adj)[None], wstate,
        agent_in_tube=win_tube, agent_active=np.array([not bool(a.status) for a in wa], dtype=bool)[None],
        agent_routes=routes, true_world_states=wtrue)
    comb = available_actions * mask[0]
    empty = comb.sum(axis=-1) == 0
    comb[empty] = available_actions[empty]          # never all-zero (as the runner)
    return comb, int(no_safe[0].sum())


# ---------------------------------------------------------------- main
def main(argv):
    dargs = distill_args(argv)
    all_args = parse_args(argv, get_config())
    all_args = modify_args(all_args.model_dir, all_args)
    all_args.use_render = True
    assert getattr(all_args, "cbf_filter", False), "the teacher needs the final filter flags (--cbf_eval --cbf_filter ...)"
    torch.manual_seed(all_args.seed); np.random.seed(all_args.seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(all_args.seed + 7919)

    envs = make_render_env(all_args)
    from onpolicy.runner.shared.graph_mpe_runner import GMPERunner as Runner
    runner = Runner({"all_args": all_args, "envs": envs, "eval_envs": None, "num_agents": all_args.num_agents,
                     "device": torch.device("cpu"), "run_dir": None})
    _attach_cbf_eval(runner, all_args, torch.device("cpu"))
    import onpolicy.algorithms.utils.cbf_eval_metrics as _cbfm
    runner.cbf_eval_scen = _cbfm.get_scenario(envs)

    teacher = runner.trainer.policy.actor            # the ORIGINAL weights (from --model_dir); never trained here
    teacher.eval()
    student = copy.deepcopy(teacher)
    if dargs.distill_student:
        student.load_state_dict(torch.load(os.path.join(dargs.distill_student, "actor.pt"), map_location="cpu"))
    student.eval()

    na, n_act = all_args.num_agents, envs.action_space[0].n
    rN, hid = runner.recurrent_N, runner.hidden_size
    stop = np.zeros(n_act, np.float32); stop[n_act // 2] = 1
    S = {k: [] for k in ("obs", "node_obs", "agent_id", "rnn", "mask", "label", "orig_logits", "interv", "w", "adj_idx", "fmask", "blocked")}
    ADJ = []
    ep_log = []
    t0 = time.time()
    for ep in range(all_args.render_episodes):
        obs, agent_id, node_obs, adj = envs.reset(ep)
        h_t = np.zeros((na, rN, hid), np.float32); h_s = np.zeros((na, rN, hid), np.float32)
        masks = np.ones((na, 1), np.float32)
        avail = np.ones((na, n_act), np.float32)
        n_int = n_kept = n_fly = loss_ps = no_safe = n_blk = 0
        min_d = np.inf
        for step in range(all_args.episode_length):
            zero = masks[:, 0] == 0
            if not zero.all():
                avail = np.ones((na, n_act), np.float32)
            avail[zero] = stop
            filt, ns = _filter_mask(runner, envs, node_obs, adj, avail.copy())
            no_safe += ns
            o, no, ad, ai = np.concatenate(obs), np.concatenate(node_obs), np.concatenate(adj), np.concatenate(agent_id)
            with torch.no_grad():
                lt, h_t_new = actor_logits(teacher, o, no, ad, ai, h_t, masks)
                ls, h_s_new = actor_logits(student, o, no, ad, ai, h_s, masks)
                a_orig = masked_argmax(lt, avail).numpy()          # original policy, no filter
                a_teach = masked_argmax(lt, filt).numpy()          # original policy + filter = label
                a_stud = masked_argmax(ls, avail).numpy()          # student alone
            fly = np.array([not bool(a.status) for a in envs.envs[0].world.agents]) & ~zero
            interv = (a_teach != a_orig) & fly
            blocked = ((filt < avail).any(axis=1)) & fly & ~interv     # filter removed moves, choice unchanged
            u = rng.random(na)
            pk = np.where(interv, 1.0, np.where(blocked, dargs.distill_keep_blocked, dargs.distill_keep_frac))
            keep = fly & (u < pk)
            if keep.any():
                ADJ.append(ad[0].astype(np.float32))                # adjacency is the same for every aircraft
                k = np.where(keep)[0]
                S["obs"].append(o[k]); S["node_obs"].append(no[k]); S["agent_id"].append(ai[k])
                S["rnn"].append(h_s[k]); S["mask"].append(masks[k]); S["label"].append(a_teach[k])
                S["orig_logits"].append(lt.numpy()[k]); S["interv"].append(interv[k])
                S["w"].append((1.0 / pk[k]).astype(np.float32))
                S["fmask"].append(filt[k].astype(np.uint8)); S["blocked"].append(blocked[k])
                S["adj_idx"].append(np.full(len(k), len(ADJ) - 1, np.int32))
            n_int += int(interv.sum()); n_blk += int(blocked.sum()); n_kept += int(keep.sum()); n_fly += int(fly.sum())
            # who flies this second: teacher (prob beta) or student, per aircraft
            act = np.where(rng.random(na) < dargs.distill_beta, a_teach, a_stud)
            act[zero] = n_act // 2
            h_t, h_s = h_t_new.numpy(), h_s_new.numpy()
            actions_env = np.eye(n_act, dtype=np.float32)[act][None]
            obs, agent_id, node_obs, adj, rewards, dones, infos, reset_count = envs.step(actions_env)
            # loss-of-separation monitor (pair-seconds, both aircraft flying) - for round-to-round checks only
            W = envs.envs[0].world.agents
            P = np.array([a.state.p_pos for a in W]); fl = np.array([not bool(a.status) for a in W])
            D = np.linalg.norm(P[:, None] - P[None], axis=2)
            loss_ps += int((np.triu(D < SEP_KM, 1) & fl[:, None] & fl[None]).sum())
            pair = np.triu(np.ones_like(D, bool), 1) & fl[:, None] & fl[None]
            if pair.any():
                min_d = min(min_d, float(D[pair].min()))
            d = dones[0]
            h_t[d] = 0; h_s[d] = 0
            masks = np.ones((na, 1), np.float32); masks[d] = 0
            if np.all(d):
                masks[:] = 1
            if reset_count > 0:
                break
        info = runner.process_infos(infos)
        _, success, _ = runner.get_fraction_episodes(info)
        ep_log.append((float(np.mean(success)), loss_ps, n_fly, n_int, n_kept, no_safe, step + 1, min_d, n_blk))
        print(f"[distill] ep {ep}: success {100 * np.mean(success):.1f}%  loss pair-seconds {loss_ps}  "
              f"flying aircraft-seconds {n_fly}  filter interventions {n_int} ({100 * n_int / max(n_fly, 1):.2f}%)  "
              f"moves removed but choice kept {n_blk} ({100 * n_blk / max(n_fly, 1):.1f}%)  kept {n_kept}  no-safe {no_safe}  steps {step + 1}  min separation {1000 * min_d:.0f} m  ({time.time() - t0:.0f} s)", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(dargs.distill_out)), exist_ok=True)
    out = {k: (np.concatenate(v) if v else np.zeros((0,))) for k, v in S.items()}
    out["adj"] = np.stack(ADJ) if ADJ else np.zeros((0, na, na), np.float32)
    out["episodes"] = np.array(ep_log, np.float64)
    out["meta"] = np.array([na, all_args.seed, dargs.distill_beta, dargs.distill_keep_frac], np.float64)
    np.savez_compressed(dargs.distill_out, **out)
    print(f"[distill] SAVED {dargs.distill_out}: {len(out['label'])} samples, "
          f"{int(out['interv'].sum()) if len(out['label']) else 0} interventions", flush=True)
    envs.close()


if __name__ == "__main__":
    main(sys.argv[1:])
