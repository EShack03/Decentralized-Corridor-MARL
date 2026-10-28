"""
Evaluate a frozen policy WITH the CBF safety filter active vs WITHOUT it
(baseline), over the SAME episodes, under the canonical repro.py evaluation
protocol, and report safety, success, conformance, separation-severity and
efficiency metrics for both.

Despite the filename, this is NOT specific to "Stage 2": --eval_tag (or the
model_dir) labels whatever policy is being evaluated (original / 3A / 3B). The
policy and the barrier network are both FROZEN here — nothing is trained.

WHAT IS MEASURED
    * collisions (= losses of separation, LoS) — agent-pair-steps within SEP
      (500 ft = 0.1524 km) per episode. This is the env's OWN collision counter
      (get_collisions), kept under the original's name "collisions"; the gloss
      "loss of separation" states what it physically means (a per-step
      separation-violation count, not distinct crashes). An independent geometric
      cross-check (coll_xchk, from world positions) is reported alongside it, and
      the count is additionally split in- vs outside-corridor (see below).
    * success rate      — fraction of agents reaching goal (phase_reached >= 2);
                          must NOT drop when the filter is added.
    * corridor conformance %  — 100*(1 - violation fraction); higher = better.
    * separation severity     — min separation, closest approach, max intrusion
                          depth, integrated intrusion, near-miss rate (within
                          NEARMISS_MULT*SEP), collision-free-episode %.
    * in/outside-corridor split — collisions, worst intrusion and min separation
                          computed separately for conflicts inside a corridor tube
                          vs in open airspace (in + out == total for the counts).
                          "In corridor" = either aircraft of the pair is inside
                          its tube (scenario's own is_in_tube test).
    * efficiency        — throughput (agents delivered to goal per minute) and
                          completion time (time-to-goal of agents that reached).
    * filter behaviour  — no-safe-action rate and avg primitives blocked.

HOW THE NUMBERS STAY CONSISTENT WITH THE ORIGINAL
    The rollout loop mirrors the runner's own eval loop and calls the runner's
    OWN metric helpers (process_infos / get_collisions / get_fraction_episodes /
    get_conformation_percentages / get_spacing_violations / get_delta_spacing),
    so collisions, success and conformance are counted exactly as the original
    density sweep counts them. --canonical_eval (default True) forces the repro.py eval
    protocol (formation_type=point, total_actions=9, zeroshift=10,
    use_dones=False) so a plain invocation matches the original methodology.
    The ONLY difference between baseline and filtered is whether the barrier
    masks the primitives before the policy acts.

MULTI-SEED
    --seeds 1,2,3 repeats every configuration over each seed (fresh env per
    seed) and reports mean +/- population std across seeds. Omit it for a
    single-seed run.

USAGE (from repo root, venv active) — single filtered config, 3 seeds:

    python onpolicy/scripts/eval_cbf_stage2.py \
        --model_dir "model_weights" \
        --env_name GraphMPE --algorithm_name rmappo \
        --scenario_name three_phase_graph_sequential \
        --dynamics_type air_taxi --num_agents 5 --num_landmarks 5 \
        --world_size 5 --episode_length 200 --render_episodes 50 \
        --n_rollout_threads 8 --seeds 1,2,3 \
        --cbf_barrier cbf_stage1_barrier.pt \
        --cbf_alpha 1.0 --cbf_horizon 3 --cbf_neighbour_model constant_velocity

    Add --cbf_density_sweep (grid at each N in --cbf_density_Ns) for a density
    sweep, or --cbf_sweep for an alpha x horizon x fallback design sweep at the
    current N.
"""

import os
import sys
import time
import datetime
from distutils.util import strtobool

sys.path.append(os.path.abspath(os.getcwd()))

from pathlib import Path

import numpy as np
import torch
import imageio

from onpolicy.config import get_config
from onpolicy.scripts.eval_mpe import make_render_env, parse_args, modify_args

from onpolicy.algorithms.utils.cbf_gnn import CBFGNN
from onpolicy.algorithms.utils.cbf_filter import CBFSafetyFilter, build_primitive_table


# air_taxi constants (mirrored from multiagent/config.AirTaxiConfig)
_KTS = 0.514444 * 0.001            # knots -> km/s
V_MIN = 60 * _KTS
V_MAX = 175 * _KTS
ACCEL_MIN = -0.001
ACCEL_MAX = 0.002
ANG_RATE_MAX = 0.075
N_ACCEL = 5
N_ANGRATE = 5
DT = 1.0
SEP = 500 * 0.0003048              # COLLISION_DISTANCE = SEPARATION_DISTANCE = 0.1524 km
MAX_EDGE = 2 * 1.60934            # COORDINATION_RANGE = 3.21868 km
NEARMISS_MULT = 2.0                # "proximity event" buffer = NEARMISS_MULT * SEP


def _t2n(x):
    return x.detach().cpu().numpy()


# ---- result-file naming convention -------------------------------------------
# One pattern for every result file, so a filename says WHAT produced it and
# sorts in workflow order:
#   <stage>_<policy>_<filter>_<scenario><N>_<episodes>ep_<seeds>_<YYMMDD>.txt
# Fields: stage  = workflow anchor (s0/s1/s2/s3A/s3B), from --run_stage
#         policy = weights slug (original/shielded-train/shaped-train), --policy_slug
#         filter = none/static/cv, derived from --cbf_neighbour_model (+ fallback)
#         scenarioN = short scenario name + agent count (e.g. seqN10)
#         episodes  = episode count, seeds = Nseed (or seed<k> for a single seed)
#         YYMMDD    = run date
# run_eval.py sets --run_stage/--policy_slug; a bare harness call falls back to
# the --eval_tag so direct invocations still get a sensible, consistent name.
_SCENARIO_SHORT = {
    "working_three_phase_graph": "single",
    "three_phase_graph_sequential": "seq",
    "three_phase_graph_merge": "merge",
    "three_phase_graph_split": "split",
}


def _filter_slug(all_args, filtered):
    """none / static / cv for the filename, from the neighbour model. A baseline
    (no-filter) file is always 'none'."""
    if not filtered:
        return "none"
    nm = getattr(all_args, "cbf_neighbour_model", "static")
    return "cv" if nm == "constant_velocity" else "static"


def _seeds_slug(all_args):
    seeds = list(getattr(all_args, "_seeds", [int(all_args.seed)]))
    return f"{len(seeds)}seed" if len(seeds) > 1 else f"seed{seeds[0]}"


def result_filename(all_args, kind, filtered=True):
    """Build a convention-compliant result filename. `kind` is a short mode tag
    (eval / sweep / density) folded into the stage field so the three save paths
    stay distinguishable; everything else comes from the run's own settings."""
    stage = getattr(all_args, "run_stage", None) or all_args.eval_tag
    policy = getattr(all_args, "policy_slug", None) or all_args.eval_tag
    scen = _SCENARIO_SHORT.get(all_args.scenario_name, all_args.scenario_name)
    N = all_args.num_agents
    eps = all_args.render_episodes
    date = datetime.datetime.now().strftime("%y%m%d")
    # density sweeps span several N, so the N field is dropped for them
    scen_field = f"{scen}" if kind == "density" else f"{scen}N{N}"
    mode = "" if kind == "eval" else f"_{kind}"
    return (f"{stage}_{policy}_{_filter_slug(all_args, filtered)}_"
            f"{scen_field}_{eps}ep_{_seeds_slug(all_args)}_{date}{mode}.txt")


def _get_scenario(envs):
    """The scenario instance behind the (single) env, reached via its own bound
    info_callback. Used only for the side-effect-free is_in_tube() query; returns
    None if the env doesn't expose it (then the corridor split is skipped)."""
    try:
        return envs.envs[0].info_callback.__self__
    except Exception:
        return None


def _in_corridor_flags(scen, world, world_agents):
    """Per-agent boolean: is this agent geometrically inside its own corridor tube
    right now? Uses the scenario's pure is_in_tube(world, pos, current_tube) (no
    state mutation). Returns None if unavailable, so callers fall back to whole-run
    (unsplit) severity."""
    if scen is None or not hasattr(scen, "is_in_tube"):
        return None
    try:
        return np.array(
            [bool(scen.is_in_tube(world, a.state.p_pos, int(scen.current_tube[a.id])))
             for a in world_agents],
            dtype=bool)
    except Exception:
        return None


class _SepAccum:
    """Accumulates pairwise separation / severity over one episode, split into
    'in-corridor' and 'outside' by whether EITHER agent of a conflicting pair is
    currently inside its tube. Every pair is visited once (strict upper triangle),
    so there is no double-counting and in + out == total for every quantity."""

    def __init__(self):
        self.closest = {"all": np.inf, "in": np.inf, "out": np.inf}
        self.intrusion = {"all": 0.0, "in": 0.0, "out": 0.0}   # sum max(0, SEP-d) (km*steps)
        self.maxintr = {"all": 0.0, "in": 0.0, "out": 0.0}     # worst single-step depth (km)
        self.los = {"all": 0, "in": 0, "out": 0}               # pair-steps with d < SEP
        self.near_pairsteps = 0                                # pair-steps with d < NEARMISS_MULT*SEP
        self.pairsteps = 0                                     # total pair-steps (near-miss denom)

    def update(self, pos, in_corr, active=None):
        """pos: (na, 2) post-step positions. in_corr: (na,) bool or None (no split).
        active: (na,) bool or None. A pair is scored ONLY if both aircraft are
        active (not yet parked at goal), matching the env's is_collision guard
        (`if agent1.status or agent2.status: return False`). A parked aircraft sitting
        on its goal is not a traffic conflict, so excluding it keeps every severity
        number reconciled with the env's collision counter."""
        if pos.shape[0] < 2:
            return
        i, j = np.triu_indices(pos.shape[0], k=1)
        d = np.linalg.norm(pos[i] - pos[j], axis=1)            # (pairs,)
        # drop pairs involving a parked (finished) aircraft, as the env does
        if active is not None:
            keep = active[i] & active[j]
            if not keep.any():
                return
            i, j, d = i[keep], j[keep], d[keep]
        depth = np.maximum(0.0, SEP - d)                       # intrusion depth per pair
        los = d < SEP
        # pair is "in-corridor" if either endpoint is inside its tube
        if in_corr is None:
            buckets = [("all", np.ones_like(d, dtype=bool))]
        else:
            pair_in = in_corr[i] | in_corr[j]
            buckets = [("all", np.ones_like(d, dtype=bool)),
                       ("in", pair_in), ("out", ~pair_in)]
        for key, sel in buckets:
            if not sel.any():
                continue
            dsel, depthsel, lossel = d[sel], depth[sel], los[sel]
            self.closest[key] = min(self.closest[key], float(dsel.min()))
            self.intrusion[key] += float(depthsel.sum())        # dt = 1
            self.maxintr[key] = max(self.maxintr[key], float(depthsel.max()))
            self.los[key] += int(lossel.sum())
        self.near_pairsteps += int((d < NEARMISS_MULT * SEP).sum())
        self.pairsteps += d.shape[0]


def rollout(runner, filt=None, render=False, tag=""):
    """
    Run `render_episodes` episodes of the frozen policy. If `filt` is given, the
    barrier mask is applied to available_actions each step (ego-only lookahead).

    If `render` is True, draw each step through the ENV's own renderer — the same
    path the runner uses (envs.render('human') for a live MPE window, or
    'rgb_array' frames saved to a GIF when --save_gifs). The display is therefore
    identical to the original algorithm's, just with the filter active. Use a
    small --render_episodes (e.g. 3) when rendering.

    Returns a dict of aggregated metrics plus filter behaviour counters.
    """
    envs = runner.envs
    num_agents = runner.num_agents
    n_act = envs.action_space[0].n
    episodes = runner.all_args.render_episodes
    ifi = float(getattr(runner.all_args, "ifi", 0.1))
    save_gifs = bool(getattr(runner.all_args, "save_gifs", False))
    gif_dir = "cbf_render"
    if render and save_gifs:
        os.makedirs(gif_dir, exist_ok=True)

    # Reseed at the start of every rollout so the baseline and every swept cell
    # see the SAME sequence of episodes (start positions, goals). This makes the
    # comparison fair and the whole run reproducible for the report.
    seed = int(runner.all_args.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    collisions_per_ep, success_per_ep = [], []
    conformance_per_ep, spacing_viol_per_ep = [], []
    delta_space_per_ep, timefrac_per_ep = [], []
    throughput_per_ep, comptime_per_ep = [], []
    # separation / severity, each tracked overall + split in/out of corridor.
    # One list per (quantity, bucket); bucket in {all, in, out}.
    BUCKETS = ("all", "in", "out")
    closest_per_ep = {b: [] for b in BUCKETS}
    intrusion_per_ep = {b: [] for b in BUCKETS}
    maxintr_per_ep = {b: [] for b in BUCKETS}
    los_pairstep_per_ep = {b: [] for b in BUCKETS}
    nearmiss_rate_per_ep, collisionfree_per_ep = [], []
    n_agent_steps = 0
    n_no_safe = 0
    n_blocked = 0

    scen = _get_scenario(envs)     # for the side-effect-free in-corridor query
    runner.trainer.prep_rollout()

    for ep in range(episodes):
        obs, agent_id, node_obs, adj = envs.reset(episodes)
        rnn_states = np.zeros((1, num_agents, runner.recurrent_N,
                               runner.hidden_size), dtype=np.float32)
        masks = np.ones((1, num_agents, 1), dtype=np.float32)
        last_infos = None
        all_frames = []
        sep = _SepAccum()        # per-episode separation / severity (overall + in/out split)

        for step in range(runner.episode_length):
            available_actions = np.ones((num_agents, n_act), dtype=np.float32)

            # --- barrier filter: mask primitives before the policy acts ---
            if filt is not None:
                world = np.array(
                    [a.state.values[:4] for a in envs.envs[0].world.agents],
                    dtype=np.float32)[None]                       # (1, na, 4)
                mask, no_safe = filt.masks_for_threads(
                    np.concatenate(node_obs)[None], np.concatenate(adj)[None], world)
                mask = mask[0]                                    # (na, n_act)
                combined = available_actions * mask
                empty = combined.sum(axis=-1) == 0
                combined[empty] = available_actions[empty]
                available_actions = combined
                n_agent_steps += num_agents
                n_no_safe += int(no_safe[0].sum())
                n_blocked += int((mask == 0).sum())

            action, rnn_states = runner.trainer.policy.act(
                np.concatenate(obs),
                np.concatenate(node_obs),
                np.concatenate(adj),
                np.concatenate(agent_id),
                np.concatenate(rnn_states),
                np.concatenate(masks),
                available_actions=available_actions,
                deterministic=True)
            action = np.array(np.split(_t2n(action), 1))
            rnn_states = np.array(np.split(_t2n(rnn_states), 1))

            # discrete -> one-hot for the env (air_taxi is Discrete)
            actions_env = np.squeeze(np.eye(n_act)[action], 2)

            obs, agent_id, node_obs, adj, rewards, dones, infos, _ = envs.step(actions_env)
            last_infos = infos

            # --- separation geometry from the POST-step world positions (the same
            # states the env just scored collisions on this step), accumulated
            # overall and split in/out of corridor by each agent's live tube state ---
            world_obj = envs.envs[0].world
            world_agents = world_obj.agents
            pos = np.array([a.state.p_pos for a in world_agents], dtype=np.float64)
            # active = not yet parked at goal; excludes finished aircraft from the
            # severity geometry, exactly as the env's is_collision guard does
            active = np.array([not bool(getattr(a, "status", False))
                               for a in world_agents], dtype=bool)
            sep.update(pos, _in_corridor_flags(scen, world_obj, world_agents), active)

            # --- original-style render (the ENV's own renderer), filter active ---
            if render:
                if save_gifs:
                    t0 = time.time()
                    image = envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    elapsed = time.time() - t0
                    if elapsed < ifi:
                        time.sleep(ifi - elapsed)
                else:
                    envs.render('human')

            if np.all(dones):
                break

        # save this episode's GIF (same imageio call the runner uses)
        if render and save_gifs and len(all_frames):
            fn = f"{gif_dir}/{tag}_N{num_agents}_ep{ep+1}.gif"
            imageio.mimsave(fn, all_frames, duration=ifi, loop=0)
            print(f"  saved {fn}", flush=True)

        # metrics for this episode, via the runner's own helpers
        env_infos = runner.process_infos(last_infos)
        collisions_per_ep.append(runner.get_collisions(env_infos))
        _, success, _ = runner.get_fraction_episodes(env_infos)
        success_per_ep.append(np.mean(success))

        # corridor conformance: the raw value is the VIOLATION fraction (steps
        # spent OUTSIDE the corridor after entering), so lower = better and the
        # reported conformance C% = 100*(1 - violation). This is the metric that
        # reveals whether the filter is pushing aircraft OUT of their corridors.
        conf_viol = runner.get_conformation_percentages(env_infos)
        conformance_per_ep.append(np.mean(conf_viol) if len(conf_viol) else 0.0)
        sv = runner.get_spacing_violations(env_infos)
        spacing_viol_per_ep.append(np.mean(sv) if len(sv) else 0.0)
        ds = runner.get_delta_spacing(env_infos)
        delta_space_per_ep.append(np.mean(ds) if len(ds) else 0.0)
        fracs, _, _ = runner.get_fraction_episodes(env_infos)
        timefrac_per_ep.append(np.mean(fracs) if len(fracs) else np.nan)

        # efficiency: throughput (goals delivered per minute) and completion time
        # (time-to-goal of the agents that actually reached goal). Read per-agent
        # by explicit key so success/time stay aligned. Success = phase_reached>=2,
        # matching get_fraction_episodes' definition (a built-in cross-check).
        # Timebase is the module-level DT (1.0 s/step), the same DT the severity
        # integration uses, so throughput and intrusion share one timebase.
        #
        # Completion time subtlety: the env records Time_req_to_goal only when an
        # agent gets within min_dist_thresh of its goal, and process_infos
        # substitutes the no-finish sentinel (episode_length*dt) when it never does.
        # phase_reached>=2 (exited the corridor) is NOT the same event, so a reached
        # agent can still carry the sentinel. We therefore average completion time
        # ONLY over agents whose recorded time is a real finish (< sentinel), so the
        # sentinel never inflates it. Throughput still counts phase_reached>=2.
        no_finish = runner.episode_length * DT      # process_infos' -1 substitution
        comp_times = []
        n_succ = 0
        for a in range(num_agents):
            pk, tk = f"agent{a}/phase_reached", f"agent{a}/time_to_goal"
            reached = (pk in env_infos and len(env_infos[pk]) and env_infos[pk][0] >= 2)
            if reached:
                n_succ += 1
                if tk in env_infos and len(env_infos[tk]):
                    t = float(env_infos[tk][0])
                    if t < no_finish - 1e-9:          # a real recorded finish, not the sentinel
                        comp_times.append(t)
        win_min = (runner.episode_length * DT) / 60.0
        throughput_per_ep.append(n_succ / win_min if win_min > 0 else 0.0)
        comptime_per_ep.append(np.mean(comp_times) if comp_times else np.nan)

        # finalise this episode's separation / severity metrics (overall + split)
        for b in BUCKETS:
            c = sep.closest[b]
            closest_per_ep[b].append(c if np.isfinite(c) else np.nan)
            intrusion_per_ep[b].append(sep.intrusion[b])
            maxintr_per_ep[b].append(sep.maxintr[b])
            los_pairstep_per_ep[b].append(sep.los[b])
        nearmiss_rate_per_ep.append(sep.near_pairsteps / sep.pairsteps if sep.pairsteps else 0.0)
        collisionfree_per_ep.append(1.0 if sep.los["all"] == 0 else 0.0)

        print(f"  ep {ep+1}/{episodes}: collisions={collisions_per_ep[-1]:.1f}  "
              f"success={success_per_ep[-1]*100:.1f}%  "
              f"conformance={100*(1-conformance_per_ep[-1]):.1f}%  "
              f"min_sep={closest_per_ep['all'][-1]:.3f}km", flush=True)

    mean_conf_viol = float(np.mean(conformance_per_ep))
    out = {
        "mean_collisions": float(np.mean(collisions_per_ep)),   # = LoS count (per-step, halved)
        "mean_success_rate": float(np.mean(success_per_ep)),
        "mean_conformance_pct": 100.0 * (1.0 - mean_conf_viol),   # higher = better
        "mean_spacing_violation_pct": 100.0 * float(np.mean(spacing_viol_per_ep)),
        "mean_delta_space": float(np.mean(delta_space_per_ep)),
        "mean_time_to_goal_frac": float(np.nanmean(timefrac_per_ep)),
        "mean_throughput": float(np.mean(throughput_per_ep)),   # agents-to-goal per minute
        "mean_completion_time_s": float(np.nanmean(comptime_per_ep)),  # of agents that reached goal
        "collisions_per_ep": collisions_per_ep,
        # --- separation / severity (overall; 'all' bucket) ---
        "mean_closest_approach_km": float(np.nanmean(closest_per_ep["all"])),
        "min_closest_approach_km": float(np.nanmin(closest_per_ep["all"])),
        "mean_integrated_intrusion_km": float(np.mean(intrusion_per_ep["all"])),
        "mean_max_intrusion_km": float(np.mean(maxintr_per_ep["all"])),
        "mean_los_pairsteps": float(np.mean(los_pairstep_per_ep["all"])),  # cross-check vs mean_collisions
        "mean_nearmiss_rate_pct": 100.0 * float(np.mean(nearmiss_rate_per_ep)),
        "collision_free_episode_pct": 100.0 * float(np.mean(collisionfree_per_ep)),
        "sep_standard_km": SEP,
        "nearmiss_mult": NEARMISS_MULT,
    }
    # --- in/outside-corridor split: LoS count, worst intrusion, closest approach.
    # When the split is available, in + out == total for LoS and intrusion (every
    # conflicting pair is bucketed exactly once). The keys are ALWAYS written so
    # downstream code never KeyErrors; when the scenario couldn't be queried the
    # buckets are empty (0 for counts/intrusion, NaN for min-sep) and
    # corridor_split_available is False so the tables simply omit the breakdown.
    have_split = scen is not None and hasattr(scen, "is_in_tube")
    for b in ("in", "out"):
        out[f"mean_los_pairsteps_{b}"] = float(np.mean(los_pairstep_per_ep[b]))
        out[f"mean_max_intrusion_km_{b}"] = float(np.mean(maxintr_per_ep[b]))
        out[f"mean_integrated_intrusion_km_{b}"] = float(np.mean(intrusion_per_ep[b]))
        cb = np.array(closest_per_ep[b], dtype=float)
        out[f"min_closest_approach_km_{b}"] = (
            float(np.nanmin(cb)) if np.isfinite(cb).any() else np.nan)
    out["corridor_split_available"] = bool(have_split)
    if filt is not None:
        out["no_safe_action_rate"] = (n_no_safe / n_agent_steps) if n_agent_steps else 0.0
        out["avg_primitives_blocked"] = (n_blocked / n_agent_steps) if n_agent_steps else 0.0
        out["agent_steps"] = n_agent_steps
    return out


def build_runner(all_args, device):
    envs = make_render_env(all_args)
    from onpolicy.runner.shared.graph_mpe_runner import GMPERunner as Runner
    config = {
        "all_args": all_args, "envs": envs, "eval_envs": envs,
        "num_agents": all_args.num_agents, "device": device, "run_dir": Path("."),
    }
    return Runner(config)


# keys that are fixed protocol constants, not per-seed measurements — carried
# through without a spurious across-seed std
_CONST_KEYS = {"sep_standard_km", "nearmiss_mult", "agent_steps"}


def _aggregate_seeds(per_seed, seeds):
    """Aggregate a list of per-seed metric dicts into mean + std across seeds.
    Scalar numeric metric keys get `k` (mean across seeds) and `k_std` (population
    std across seeds); protocol constants (_CONST_KEYS) and list-valued keys are
    carried through from the first seed, with no std."""
    if len(per_seed) == 1:
        out = dict(per_seed[0])
        out["n_seeds"] = 1
        out["seeds"] = list(seeds)
        return out
    out = {}
    for k, v in per_seed[0].items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or k in _CONST_KEYS:
            out[k] = v                      # carry constants / lists / flags through
            continue
        vals = np.array([p.get(k, np.nan) for p in per_seed], dtype=float)
        out[k] = float(np.nanmean(vals))
        out[k + "_std"] = float(np.nanstd(vals))
    out["n_seeds"] = len(per_seed)
    out["seeds"] = list(seeds)
    return out


def eval_config(all_args, device, filt, render, tag, seeds):
    """Run one configuration over every seed (fresh runner/env per seed, so the
    env RNG is correctly re-seeded each time), then aggregate mean +/- std.
    Renders only on the first seed to avoid duplicate GIFs."""
    base_seed = int(all_args.seed)
    per_seed = []
    for i, sd in enumerate(seeds):
        all_args.seed = int(sd)
        runner = build_runner(all_args, device)
        o = rollout(runner, filt=filt, render=(render and i == 0),
                    tag=(f"{tag}_seed{sd}" if render else tag))
        per_seed.append(o)
    all_args.seed = base_seed
    return _aggregate_seeds(per_seed, seeds)


def _make_filter(barrier, prim_table, all_args, device, alpha, horizon, fallback):
    return CBFSafetyFilter(
        barrier=barrier, prim_table=prim_table, sep=SEP, dt=DT,
        v_min=V_MIN, v_max=V_MAX, alpha=alpha, eps=all_args.cbf_eps,
        horizon=horizon, fallback=fallback,
        num_agents=all_args.num_agents,
        neighbour_model=all_args.cbf_neighbour_model, device=device)


def main(argv):
    parser = get_config()
    parser.add_argument("--cbf_barrier", type=str, default="cbf_stage1_barrier.pt")
    parser.add_argument("--cbf_alpha", type=float, default=1.0)
    parser.add_argument("--cbf_eps", type=float, default=0.0)
    parser.add_argument("--cbf_horizon", type=int, default=3,
                        help="steps ahead the filter rolls each primitive (>=1)")
    parser.add_argument("--cbf_fallback", type=str, default="least_unsafe",
                        choices=["least_unsafe", "unfiltered"],
                        help="what the filter does when no primitive is safe")
    parser.add_argument("--cbf_neighbour_model", type=str, default="static",
                        choices=["static", "constant_velocity"],
                        help="how neighbours are propagated in the filter's "
                        "lookahead: static (held fixed) or constant_velocity "
                        "(advanced at current heading/speed)")
    parser.add_argument("--with_baseline", action="store_true",
                        help="also run the no-filter BASELINE alongside the "
                        "filtered config and report the paired A/B comparison "
                        "(change %, verdict, in/out split vs baseline). Off by "
                        "default: a plain run evaluates ONLY the filtered system "
                        "and writes a single-column result. Ignored for sweeps, "
                        "which always compute their own baseline per density.")
    parser.add_argument("--cbf_render", action="store_true",
                        help="visualise episodes via the ENV's own renderer "
                        "(original algorithm display), filter active. Live window "
                        "by default; with --save_gifs saves a GIF per episode to "
                        "cbf_render/. Single-config only (ignored in sweeps). Use "
                        "a small --render_episodes (e.g. 3).")
    parser.add_argument("--cbf_hidden", type=int, default=64)
    parser.add_argument("--cbf_layers", type=int, default=2)
    parser.add_argument("--cbf_sweep", action="store_true",
                        help="sweep alpha x horizon x fallback instead of one config")
    parser.add_argument("--cbf_sweep_alphas", type=str, default="0.1,0.3,0.5,1.0")
    parser.add_argument("--cbf_sweep_horizons", type=str, default="1,3,5")
    parser.add_argument("--cbf_sweep_fallbacks", type=str, default="least_unsafe")
    parser.add_argument("--cbf_density_sweep", action="store_true",
                        help="run the full alpha x horizon grid at each density N "
                             "in --cbf_density_Ns (num_landmarks tracks N)")
    parser.add_argument("--cbf_density_Ns", type=str, default="5,8,10")
    parser.add_argument("--canonical_eval", type=lambda x: bool(strtobool(x)),
                        default=True,
                        help="force the canonical repro.py eval protocol "
                        "(formation_type=point, total_actions=9, zeroshift=10, "
                        "use_dones=False). Default True so every run is consistent "
                        "with the original methodology; set False to opt out.")
    parser.add_argument("--eval_tag", type=str, default=None,
                        help="label for output filenames / table titles "
                        "(e.g. original, 3A, 3B). If omitted it is derived from "
                        "--model_dir (model_weights -> 'original', else the "
                        "experiment folder name).")
    parser.add_argument("--seeds", type=str, default=None,
                        help="comma-separated seeds to repeat every config over "
                        "(e.g. 1,2,3). Each config is run once per seed and reported "
                        "as mean +/- std across seeds. If omitted, only --seed is "
                        "used (single-seed run, no error bars).")
    parser.add_argument("--num_seeds", type=int, default=None,
                        help="shorthand for --seeds: use this many seeds counting "
                        "up from --seed (e.g. --seed 1 --num_seeds 2 -> seeds 1,2). "
                        "Any positive count is allowed. Ignored if --seeds is given.")
    # naming-convention passthroughs (normally set by run_eval.py; a bare harness
    # call falls back to --eval_tag so direct runs still get a consistent name).
    parser.add_argument("--run_stage", type=str, default=None,
                        help="workflow stage tag for the result filename "
                        "(s0/s1/s2/s3A/s3B). Defaults to --eval_tag.")
    parser.add_argument("--policy_slug", type=str, default=None,
                        help="policy-weights slug for the result filename "
                        "(original/shielded-train/shaped-train). Defaults to --eval_tag.")
    all_args = parse_args(argv, parser)
    all_args = modify_args(all_args.model_dir, all_args)
    all_args.use_render = True

    # Derive a descriptive tag from the model_dir if the user didn't give one,
    # so outputs are named by WHAT was evaluated (not a misleading "stage2").
    if not getattr(all_args, "eval_tag", None):
        parts = os.path.normpath(all_args.model_dir).replace("\\", "/").split("/")
        if "model_weights" in parts:
            all_args.eval_tag = "original"
        elif "rmappo" in parts and parts.index("rmappo") + 1 < len(parts):
            all_args.eval_tag = parts[parts.index("rmappo") + 1]   # experiment name
        else:
            all_args.eval_tag = os.path.basename(os.path.normpath(all_args.model_dir)) or "eval"

    # ---- enforce the canonical evaluation protocol (repro.py) ----
    # modify_args keeps these from the CLI/defaults rather than config.yaml, and
    # the parser defaults are NON-canonical (formation_type=random, total_actions=5,
    # zeroshift=5). repro.py passes the canonical values explicitly; we force them
    # here so a plain invocation matches the original methodology by default.
    if getattr(all_args, "canonical_eval", True):
        canon = dict(formation_type="point", total_actions=9, zeroshift=10,
                     use_dones=False)
        for k, v in canon.items():
            setattr(all_args, k, v)
        print("[canonical eval] forced protocol: formation_type=point, "
              "total_actions=9, zeroshift=10, use_dones=False  "
              f"(episode_length={all_args.episode_length}, seed={all_args.seed})")

    # ---- resolve the seed list (error bars across seeds) ----
    # Explicit --seeds wins; else --num_seeds counts up from --seed; else single.
    if getattr(all_args, "seeds", None):
        SEEDS = [int(x) for x in all_args.seeds.split(",") if x.strip() != ""]
    elif getattr(all_args, "num_seeds", None):
        n = max(1, int(all_args.num_seeds))
        SEEDS = [int(all_args.seed) + i for i in range(n)]
    else:
        SEEDS = [int(all_args.seed)]
    all_args._seeds = SEEDS
    if len(SEEDS) > 1:
        print(f"[multi-seed] every config repeated over seeds={SEEDS}; "
              f"reported as mean +/- std across {len(SEEDS)} seeds")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)

    prim_table = build_primitive_table(N_ACCEL, N_ANGRATE, ACCEL_MIN, ACCEL_MAX, ANG_RATE_MAX)
    barrier = CBFGNN(node_obs_dim=7, edge_dim=1, hidden_size=all_args.cbf_hidden,
                     num_layers=all_args.cbf_layers, max_edge_dist=MAX_EDGE).to(device)
    barrier.load_state_dict(torch.load(all_args.cbf_barrier, map_location=device))
    barrier.eval()
    print(f"loaded barrier from {all_args.cbf_barrier}")

    alphas = [float(x) for x in all_args.cbf_sweep_alphas.split(",")]
    horizons = [int(x) for x in all_args.cbf_sweep_horizons.split(",")]
    fallbacks = [x.strip() for x in all_args.cbf_sweep_fallbacks.split(",")]

    # ---- DENSITY SWEEP: full grid at each N, one combined file ----
    if all_args.cbf_density_sweep:
        Ns = [int(x) for x in all_args.cbf_density_Ns.split(",")]
        out_path = result_filename(all_args, "density", filtered=True)
        print(f"\n### DENSITY SWEEP over N={Ns} (episodes={all_args.render_episodes}, "
              f"seed={all_args.seed}) -> {out_path}")
        with open(out_path, "w", encoding="utf-8") as fout:
            fout.write(f"# CBF density sweep [{all_args.eval_tag}, "
                       f"neighbour={all_args.cbf_neighbour_model}]  {datetime.datetime.now()}\n")
            fout.write(f"# Ns={Ns}  episodes={all_args.render_episodes}  seed={all_args.seed}  "
                       f"barrier={all_args.cbf_barrier}\n")
            fout.write(f"# alphas={alphas} horizons={horizons} fallbacks={fallbacks} "
                       f"neighbour_model={all_args.cbf_neighbour_model}\n")
            for N in Ns:
                all_args.num_agents = N
                all_args.num_landmarks = N     # landmarks track agents (repro.py convention)
                print(f"\n########## DENSITY N={N} ##########")
                base_N = eval_config(all_args, device, None, False, "baseline", SEEDS)
                print(f"baseline @N={N}: collisions={base_N['mean_collisions']:.2f}  "
                      f"success={base_N['mean_success_rate']*100:.1f}%")
                _sweep_and_report(all_args, barrier, prim_table, device, base_N,
                                  alphas, horizons, fallbacks, fout=fout)
                fout.flush()
        print(f"\n### DENSITY SWEEP complete -> {out_path}")
        return

    # render only in the single-config path (not in sweeps)
    do_render = bool(getattr(all_args, "cbf_render", False)) and not all_args.cbf_sweep

    # A sweep always needs a baseline (every cell is reported vs it); a plain
    # single-config run only computes the baseline when --with_baseline is set.
    want_baseline = all_args.cbf_sweep or bool(getattr(all_args, "with_baseline", False))

    base = None
    if want_baseline:
        # ---- baseline (no filter), over all seeds ----
        print("\n=== BASELINE (no filter) ===")
        base = eval_config(all_args, device, None, do_render, "baseline", SEEDS)
        print(f"baseline: collisions={base['mean_collisions']:.3f}  "
              f"success={base['mean_success_rate']*100:.2f}%")

    if not all_args.cbf_sweep:
        # ---- single config (filtered, or raw baseline when fallback=unfiltered) ---
        # fallback="unfiltered" means the filter never blocks anything, so the
        # policy runs raw. The driver uses this for --stage original / --filter
        # none, so print an honest banner instead of "FILTERED" in that case.
        raw = (all_args.cbf_fallback == "unfiltered")
        print("\n=== BASELINE (no filter) ===" if raw
              else "\n=== FILTERED (CBF safety filter on) ===")
        filt = _make_filter(barrier, prim_table, all_args, device,
                            all_args.cbf_alpha, all_args.cbf_horizon,
                            all_args.cbf_fallback)
        f = eval_config(all_args, device, filt, do_render, "filtered", SEEDS)
        if base is not None:
            # paired A/B comparison (baseline vs filtered)
            _print_single(all_args, base, f)
            _save_single_txt(all_args, base, f)
        else:
            # filtered-only: just the final system, single-column output
            _print_filtered_only(all_args, f)
            _save_filtered_only(all_args, f)
        return base, f

    # ---- one-density sweep (baseline already computed above) ----
    _sweep_and_report(all_args, barrier, prim_table, device,
                      base, alphas, horizons, fallbacks)
    return base, None


HDR = (f"{'alpha':>6}{'horizon':>8}{'fallback':>14}{'collisions':>12}"
       f"{'d_coll%':>10}{'success%':>10}{'conf%':>8}{'nosafe%':>9}")


def _seed_note(d):
    """One-line seed descriptor for titles ('seed=1' or 'seeds=[1,2,3]')."""
    n = d.get('n_seeds', 1)
    if n > 1:
        return f"seeds={d.get('seeds')}"
    return f"seed={d.get('seeds', ['?'])[0] if d.get('seeds') else '?'}"


def _sweep_and_report(all_args, barrier, prim_table, device, base,
                      alphas, horizons, fallbacks, fout=None):
    """
    Run the full alpha x horizon x fallback grid for the CURRENT all_args
    (i.e. the current density N), print the table, and — if `fout` is given —
    append the same table to that open file handle. `base` is the no-filter
    result for this density. Returns (rows, best_line).
    """
    seeds = list(getattr(all_args, "_seeds", [int(all_args.seed)]))
    print(f"\n=== SWEEP N={all_args.num_agents}: alphas={alphas} "
          f"horizons={horizons} fallbacks={fallbacks} seeds={seeds} ===")
    rows = []
    for fb in fallbacks:
        for hz in horizons:
            for al in alphas:
                print(f"\n--- N={all_args.num_agents}, alpha={al}, horizon={hz}, fallback={fb} ---")
                filt = _make_filter(barrier, prim_table, all_args, device, al, hz, fb)
                r = eval_config(all_args, device, filt, False,
                                f"a{al}_h{hz}_{fb}", seeds)
                rows.append((al, hz, fb, r))

    bc = base['mean_collisions']
    n_seeds = base.get('n_seeds', 1)
    seed_lbl = (f"seeds={base.get('seeds', [all_args.seed])}" if n_seeds > 1
                else f"seed={all_args.seed}")
    title = (f"CBF EVAL SWEEP [{all_args.eval_tag}, "
             f"neighbour={all_args.cbf_neighbour_model}]  (N={all_args.num_agents}, "
             f"episodes={all_args.render_episodes}, {seed_lbl})")
    baseline_line = (f"baseline (no filter): collisions={base['mean_collisions']:.2f}, "
                     f"success={base['mean_success_rate']*100:.1f}%, "
                     f"conformance={base['mean_conformance_pct']:.1f}%  "
                     f"[collisions = losses of separation (LoS): agent-pair-steps within "
                     f"{base['sep_standard_km']:.4f} km (500 ft), per episode]")

    def emit(line):
        print(line)
        if fout:
            fout.write(line + "\n")

    emit("\n" + "=" * 88)
    emit(title)
    emit(baseline_line)
    if n_seeds > 1:
        emit(f"  (mean over {n_seeds} seeds; baseline collisions std "
             f"+/-{base.get('mean_collisions_std', 0.0):.2f}, "
             f"success std +/-{base.get('mean_success_rate_std', 0.0)*100:.1f} pts)")
    emit("=" * 88)
    emit(HDR)
    emit("-" * 88)
    for (al, hz, fb, r) in rows:
        dc = 100 * (r['mean_collisions'] - bc) / max(bc, 1e-9)
        emit(f"{al:>6.2f}{hz:>8d}{fb:>14}{r['mean_collisions']:>12.2f}"
             f"{dc:>+10.1f}{r['mean_success_rate']*100:>10.1f}"
             f"{r['mean_conformance_pct']:>8.1f}{r['no_safe_action_rate']*100:>9.1f}")
    emit("-" * 88)

    # --- per-seed spread for the headline metrics (only when multi-seed) ---
    if n_seeds > 1:
        emit("")
        emit(f"ACROSS-SEED STD  (+/- population std over {n_seeds} seeds "
             f"{base.get('seeds')}; same metric, same episodes per seed)")
        emit(f"{'config':>22}{'coll_std':>10}{'succ%_std':>12}{'conf%_std':>12}")
        emit("-" * 56)
        emit(f"{'baseline(no filter)':>22}{base.get('mean_collisions_std',0.0):>10.2f}"
             f"{base.get('mean_success_rate_std',0.0)*100:>12.2f}"
             f"{base.get('mean_conformance_pct_std',0.0):>12.2f}")
        for (al, hz, fb, r) in rows:
            emit(f"{('a'+str(al)+'_h'+str(hz)+'_'+fb):>22}"
                 f"{r.get('mean_collisions_std',0.0):>10.2f}"
                 f"{r.get('mean_success_rate_std',0.0)*100:>12.2f}"
                 f"{r.get('mean_conformance_pct_std',0.0):>12.2f}")
        emit("-" * 56)

    # --- separation / severity block (richer safety view; same episodes) ---
    emit("")
    emit(f"SEPARATION / SEVERITY  (SEP={base['sep_standard_km']:.4f} km, "
         f"near-miss buffer = {base['nearmiss_mult']:.0f}x SEP). "
         f"coll_xchk should match collisions (independent cross-check).")
    emit(f"{'config':>22}{'coll':>8}{'coll_xchk':>10}{'minSep_km':>11}"
         f"{'maxIntr_km':>12}{'nearmiss%':>11}{'CFep%':>8}{'thru/min':>10}")
    emit("-" * 92)

    def _sev(label, r):
        emit(f"{label:>22}{r['mean_collisions']:>8.2f}{r['mean_los_pairsteps']:>10.2f}"
             f"{r['min_closest_approach_km']:>11.3f}"
             f"{r['mean_max_intrusion_km']:>12.3f}{r['mean_nearmiss_rate_pct']:>11.2f}"
             f"{r['collision_free_episode_pct']:>8.1f}{r['mean_throughput']:>10.3f}")
    _sev("baseline(no filter)", base)
    for (al, hz, fb, r) in rows:
        _sev(f"a{al}_h{hz}_{fb}", r)
    emit("-" * 92)

    # --- in- vs outside-corridor breakdown (where do the violations happen, and
    # does the filter cut severity more inside the lanes or in open airspace?) ---
    if base.get("corridor_split_available", False):
        emit("")
        emit("IN- vs OUTSIDE-CORRIDOR collisions & worst intrusion  "
             "(in-corridor = either aircraft inside its tube; coll_in + coll_out = collisions)")
        emit(f"{'config':>22}{'coll_in':>9}{'coll_out':>9}"
             f"{'maxIntr_in':>12}{'maxIntr_out':>13}")
        emit("-" * 65)

        def _split(label, r):
            emit(f"{label:>22}{r['mean_los_pairsteps_in']:>9.2f}"
                 f"{r['mean_los_pairsteps_out']:>9.2f}"
                 f"{r['mean_max_intrusion_km_in']:>12.3f}"
                 f"{r['mean_max_intrusion_km_out']:>13.3f}")
        _split("baseline(no filter)", base)
        for (al, hz, fb, r) in rows:
            _split(f"a{al}_h{hz}_{fb}", r)
        emit("-" * 65)

    ok = [(al, hz, fb, r) for (al, hz, fb, r) in rows
          if r['mean_success_rate'] >= base['mean_success_rate'] - 0.03]
    best_line = None
    if ok:
        b_al, b_hz, b_fb, b_r = min(ok, key=lambda x: x[3]['mean_collisions'])
        best_line = (f"BEST @N={all_args.num_agents} (success within 3 pts): "
                     f"alpha={b_al}, horizon={b_hz}, fallback={b_fb} -> "
                     f"collisions {bc:.2f} -> {b_r['mean_collisions']:.2f} "
                     f"({100*(b_r['mean_collisions']-bc)/max(bc,1e-9):+.1f}%), "
                     f"no-safe {b_r['no_safe_action_rate']*100:.1f}%")
        emit(best_line)

    # standalone single-density save (only when not part of a density sweep file)
    if fout is None:
        out_path = result_filename(all_args, "sweep", filtered=True)
        with open(out_path, "w", encoding="utf-8") as f2:
            f2.write(f"# CBF safety-filter sweep [{all_args.eval_tag}, "
                     f"neighbour={all_args.cbf_neighbour_model}]  {datetime.datetime.now()}\n")
            f2.write(f"# barrier={all_args.cbf_barrier}\n\n")
            f2.write(baseline_line + "\n\n" + HDR + "\n" + "-" * 88 + "\n")
            for (al, hz, fb, r) in rows:
                dc = 100 * (r['mean_collisions'] - bc) / max(bc, 1e-9)
                f2.write(f"{al:>6.2f}{hz:>8d}{fb:>14}{r['mean_collisions']:>12.2f}"
                         f"{dc:>+10.1f}{r['mean_success_rate']*100:>10.1f}"
                         f"{r['mean_conformance_pct']:>8.1f}{r['no_safe_action_rate']*100:>9.1f}\n")
            if best_line:
                f2.write("\n" + best_line + "\n")
        print(f"\nsaved sweep results -> {out_path}")

    return rows, best_line


def _filtered_only_lines(all_args, f):
    """The metric rows for a single (filtered) configuration, no baseline column.
    Shared by the on-screen print and the saved file so they never drift."""
    L = []
    L.append(f"{'metric':<30}{'value':>16}")
    L.append(f"{'collisions (LoS)':<30}{f['mean_collisions']:>16.3f}")
    L.append(f"{'  collisions cross-check':<30}{f['mean_los_pairsteps']:>16.3f}")
    L.append(f"{'mean success rate':<30}{f['mean_success_rate']*100:>15.2f}%")
    L.append(f"{'corridor conformance':<30}{f['mean_conformance_pct']:>15.2f}%")
    L.append(f"{'collision-free episodes':<30}{f['collision_free_episode_pct']:>15.1f}%")
    L.append(f"{'min separation (km)':<30}{f['min_closest_approach_km']:>16.3f}")
    L.append(f"{'mean closest approach (km)':<30}{f['mean_closest_approach_km']:>16.3f}")
    L.append(f"{'max intrusion depth (km)':<30}{f['mean_max_intrusion_km']:>16.3f}")
    L.append(f"{'integrated intrusion (km.st)':<30}{f['mean_integrated_intrusion_km']:>16.3f}")
    L.append(f"{'near-miss rate %':<30}{f['mean_nearmiss_rate_pct']:>16.2f}")
    L.append(f"{'spacing violations %':<30}{f['mean_spacing_violation_pct']:>16.2f}")
    L.append(f"{'throughput (agents/min)':<30}{f['mean_throughput']:>16.3f}")
    L.append(f"{'completion time (s, reached)':<30}{f['mean_completion_time_s']:>16.2f}")
    # in/outside-corridor split (single column) if the scenario exposed it
    if f.get("corridor_split_available", False):
        L.append("")
        L.append("IN- vs OUTSIDE-CORRIDOR  (in-corridor = either aircraft inside its tube;"
                 " in + out = total)")
        L.append(f"{'collisions in corridor':<30}{f['mean_los_pairsteps_in']:>16.3f}")
        L.append(f"{'collisions outside corridor':<30}{f['mean_los_pairsteps_out']:>16.3f}")
        L.append(f"{'max intrusion in (km)':<30}{f['mean_max_intrusion_km_in']:>16.3f}")
        L.append(f"{'max intrusion out (km)':<30}{f['mean_max_intrusion_km_out']:>16.3f}")
    return L


def _is_raw(all_args):
    """True when the 'filter' never blocks anything (fallback=unfiltered), i.e.
    this is really the raw baseline policy, not a filtered system. The driver
    sets fallback=unfiltered for --stage original / --filter none."""
    return getattr(all_args, "cbf_fallback", "least_unsafe") == "unfiltered"


def _print_filtered_only(all_args, f):
    raw = _is_raw(all_args)
    kind = "BASELINE" if raw else "FILTERED SYSTEM"
    print("\n" + "=" * 62)
    seedtag = _seed_note(f)
    if raw:
        print(f"CBF EVAL RESULT [{all_args.eval_tag}] - {kind}  "
              f"(N={all_args.num_agents}, episodes={all_args.render_episodes}, {seedtag}, "
              f"no filter / raw policy)")
        print("  (raw baseline policy, no safety filter active)")
    else:
        print(f"CBF EVAL RESULT [{all_args.eval_tag}] - {kind}  "
              f"(N={all_args.num_agents}, episodes={all_args.render_episodes}, {seedtag}, "
              f"alpha={all_args.cbf_alpha}, horizon={all_args.cbf_horizon}, "
              f"fallback={all_args.cbf_fallback}, neighbour={all_args.cbf_neighbour_model})")
        print("  (filtered system only; run with --with_baseline for the A/B comparison)")
    print("=" * 62)
    if f.get('n_seeds', 1) > 1:
        print(f"(mean over {f['n_seeds']} seeds {f.get('seeds')})")
    for line in _filtered_only_lines(all_args, f):
        print(line)
    # "Filter behaviour" only means something when a filter is actually acting
    if not raw:
        print("\nFilter behaviour:")
        print(f"  no-safe-action rate    : {f['no_safe_action_rate']*100:.2f}%  "
              f"(of {f['agent_steps']} agent-steps)")
        print(f"  avg primitives blocked : {f['avg_primitives_blocked']:.2f} / 25 per step")


def _save_filtered_only(all_args, f):
    raw = _is_raw(all_args)
    out_path = result_filename(all_args, "eval", filtered=not raw)
    with open(out_path, "w", encoding="utf-8") as fout:
        if raw:
            fout.write(f"# BASELINE result [{all_args.eval_tag}]  {datetime.datetime.now()}\n")
            fout.write("# RAW BASELINE POLICY (no safety filter active)\n")
            fout.write(f"# N={all_args.num_agents}  episodes={all_args.render_episodes}  "
                       f"{_seed_note(f)}  barrier=n/a (no filter)\n\n")
        else:
            fout.write(f"# CBF safety-filter result [{all_args.eval_tag}, "
                       f"neighbour={all_args.cbf_neighbour_model}]  {datetime.datetime.now()}\n")
            fout.write("# FILTERED SYSTEM ONLY (no baseline; run --with_baseline for A/B)\n")
            fout.write(f"# N={all_args.num_agents}  episodes={all_args.render_episodes}  "
                       f"{_seed_note(f)}  "
                       f"alpha={all_args.cbf_alpha}  horizon={all_args.cbf_horizon}  "
                       f"fallback={all_args.cbf_fallback}  "
                       f"neighbour_model={all_args.cbf_neighbour_model}  "
                       f"barrier={all_args.cbf_barrier}\n\n")
        fout.write(f"# collisions = losses of separation (LoS): agent-pair-steps within "
                   f"{f['sep_standard_km']:.4f} km (500 ft), per episode\n\n")
        if f.get('n_seeds', 1) > 1:
            fout.write(f"# values are means over {f['n_seeds']} seeds {f.get('seeds')}\n\n")
        for line in _filtered_only_lines(all_args, f):
            fout.write(line + "\n")
        fout.write("\n")
        if not raw:
            fout.write(f"no-safe-action rate    : {f['no_safe_action_rate']*100:.2f}%  "
                       f"(of {f['agent_steps']} agent-steps)\n")
            fout.write(f"avg primitives blocked : {f['avg_primitives_blocked']:.2f} / 25 per step\n")
        if f.get('n_seeds', 1) > 1:
            fout.write(f"\n# ACROSS-SEED STD (+/- population std over "
                       f"{f['n_seeds']} seeds {f.get('seeds')})\n")
            fout.write(f"{'metric':<30}{'std':>16}\n")
            for lbl, k, sc in [("collisions", 'mean_collisions', 1.0),
                               ("success % ", 'mean_success_rate', 100.0),
                               ("conformance %", 'mean_conformance_pct', 1.0),
                               ("min sep km", 'min_closest_approach_km', 1.0),
                               ("throughput/min", 'mean_throughput', 1.0)]:
                fout.write(f"{lbl:<30}{f.get(k+'_std',0.0)*sc:>16.3f}\n")
    print(f"\nsaved result -> {out_path}")


def _save_single_txt(all_args, base, f):
    out_path = result_filename(all_args, "eval", filtered=True)
    dc = f['mean_collisions'] - base['mean_collisions']
    ds = (f['mean_success_rate'] - base['mean_success_rate']) * 100
    with open(out_path, "w", encoding="utf-8") as fout:
        fout.write(f"# CBF safety-filter result [{all_args.eval_tag}, "
                   f"neighbour={all_args.cbf_neighbour_model}]  {datetime.datetime.now()}\n")
        fout.write(f"# N={all_args.num_agents}  episodes={all_args.render_episodes}  "
                   f"{_seed_note(base)}  "
                   f"alpha={all_args.cbf_alpha}  horizon={all_args.cbf_horizon}  "
                   f"fallback={all_args.cbf_fallback}  "
                   f"neighbour_model={all_args.cbf_neighbour_model}  "
                   f"barrier={all_args.cbf_barrier}\n\n")
        fout.write(f"# collisions = losses of separation (LoS): agent-pair-steps within "
                   f"{base['sep_standard_km']:.4f} km (500 ft), per episode\n\n")
        if base.get('n_seeds', 1) > 1:
            fout.write(f"# values are means over {base['n_seeds']} seeds "
                       f"{base.get('seeds')}; see the ACROSS-SEED STD block below\n\n")
        fout.write(f"{'metric':<30}{'baseline':>16}{'filtered':>16}\n")
        fout.write(f"{'collisions (LoS)':<30}{base['mean_collisions']:>16.3f}"
                   f"{f['mean_collisions']:>16.3f}\n")
        fout.write(f"{'  collisions cross-check':<30}{base['mean_los_pairsteps']:>16.3f}"
                   f"{f['mean_los_pairsteps']:>16.3f}\n")
        fout.write(f"{'mean success rate':<30}{base['mean_success_rate']*100:>15.2f}%"
                   f"{f['mean_success_rate']*100:>15.2f}%\n")
        fout.write(f"{'corridor conformance':<30}{base['mean_conformance_pct']:>15.2f}%"
                   f"{f['mean_conformance_pct']:>15.2f}%\n")
        fout.write(f"{'collision-free episodes':<30}{base['collision_free_episode_pct']:>15.1f}%"
                   f"{f['collision_free_episode_pct']:>15.1f}%\n")
        fout.write(f"{'min separation (km)':<30}{base['min_closest_approach_km']:>16.3f}"
                   f"{f['min_closest_approach_km']:>16.3f}\n")
        fout.write(f"{'mean closest approach (km)':<30}{base['mean_closest_approach_km']:>16.3f}"
                   f"{f['mean_closest_approach_km']:>16.3f}\n")
        fout.write(f"{'max intrusion depth (km)':<30}{base['mean_max_intrusion_km']:>16.3f}"
                   f"{f['mean_max_intrusion_km']:>16.3f}\n")
        fout.write(f"{'integrated intrusion (km.st)':<30}{base['mean_integrated_intrusion_km']:>16.3f}"
                   f"{f['mean_integrated_intrusion_km']:>16.3f}\n")
        fout.write(f"{'near-miss rate %':<30}{base['mean_nearmiss_rate_pct']:>16.2f}"
                   f"{f['mean_nearmiss_rate_pct']:>16.2f}\n")
        fout.write(f"{'spacing violations %':<30}{base['mean_spacing_violation_pct']:>16.2f}"
                   f"{f['mean_spacing_violation_pct']:>16.2f}\n")
        fout.write(f"{'throughput (agents/min)':<30}{base['mean_throughput']:>16.3f}"
                   f"{f['mean_throughput']:>16.3f}\n")
        fout.write(f"{'completion time (s, reached)':<30}{base['mean_completion_time_s']:>16.2f}"
                   f"{f['mean_completion_time_s']:>16.2f}\n")
        for line in _corridor_split_lines(base, f):
            fout.write(line + "\n")
        fout.write("\n")
        dconf = f['mean_conformance_pct'] - base['mean_conformance_pct']
        fout.write(f"change: collisions {dc:+.3f} "
                   f"({100*dc/max(base['mean_collisions'],1e-9):+.1f}%),  "
                   f"success {ds:+.2f} pts,  conformance {dconf:+.2f} pts\n")
        fout.write(f"no-safe-action rate    : {f['no_safe_action_rate']*100:.2f}%  "
                   f"(of {f['agent_steps']} agent-steps)\n")
        fout.write(f"avg primitives blocked : {f['avg_primitives_blocked']:.2f} / 25 per step\n")
        if base.get('n_seeds', 1) > 1:
            fout.write(f"\n# ACROSS-SEED STD (+/- population std over "
                       f"{base['n_seeds']} seeds {base.get('seeds')})\n")
            fout.write(f"{'metric':<30}{'base_std':>14}{'filt_std':>14}\n")
            for lbl, k, sc in [("collisions", 'mean_collisions', 1.0),
                               ("success % ", 'mean_success_rate', 100.0),
                               ("conformance %", 'mean_conformance_pct', 1.0),
                               ("min sep km", 'min_closest_approach_km', 1.0),
                               ("throughput/min", 'mean_throughput', 1.0)]:
                fout.write(f"{lbl:<30}{base.get(k+'_std',0.0)*sc:>14.3f}"
                           f"{f.get(k+'_std',0.0)*sc:>14.3f}\n")
    print(f"\nsaved result -> {out_path}")


def _corridor_split_lines(base, f):
    """Lines for the in- vs outside-corridor breakdown of LoS and intrusion
    (baseline vs filtered). Returns [] if the split wasn't available. 'in-corridor'
    = a conflicting pair with at least one aircraft inside its tube."""
    if not base.get("corridor_split_available", False):
        return []
    L = []
    L.append("")
    L.append("IN- vs OUTSIDE-CORRIDOR  (in-corridor = either aircraft inside its tube;"
             " in + out = total)")
    L.append(f"{'metric':<30}{'baseline':>16}{'filtered':>16}")
    def row(lbl, k):
        return f"{lbl:<30}{base[k]:>16.3f}{f[k]:>16.3f}"
    L.append(row("collisions in corridor",      "mean_los_pairsteps_in"))
    L.append(row("collisions outside corridor", "mean_los_pairsteps_out"))
    L.append(row("max intrusion in (km)",  "mean_max_intrusion_km_in"))
    L.append(row("max intrusion out (km)", "mean_max_intrusion_km_out"))
    # min separation can be NaN for a bucket that never had a conflict -> show n/a
    for lbl, k in [("min sep in corridor (km)", "min_closest_approach_km_in"),
                   ("min sep outside (km)",     "min_closest_approach_km_out")]:
        bv = base[k]; fv = f[k]
        bs = "    n/a" if not np.isfinite(bv) else f"{bv:16.3f}"
        fs = "    n/a" if not np.isfinite(fv) else f"{fv:16.3f}"
        L.append(f"{lbl:<30}{bs:>16}{fs:>16}")
    return L


def _print_corridor_split(base, f):
    for line in _corridor_split_lines(base, f):
        print(line)


def _print_single(all_args, base, f):
    print("\n" + "=" * 62)
    print(f"CBF EVAL RESULT [{all_args.eval_tag}]  (N={all_args.num_agents}, "
          f"episodes={all_args.render_episodes}, {_seed_note(base)}, "
          f"alpha={all_args.cbf_alpha}, horizon={all_args.cbf_horizon}, "
          f"fallback={all_args.cbf_fallback}, neighbour={all_args.cbf_neighbour_model})")
    print("=" * 62)
    if base.get('n_seeds', 1) > 1:
        print(f"(mean over {base['n_seeds']} seeds; +/- = population std across seeds)")
        print(f"{'metric':<30}{'baseline':>18}{'filtered':>18}")
        def _ms(d, k, scale=1.0, pct=False):
            m = d[k] * scale
            s = d.get(k + '_std', 0.0) * scale
            tag = '%' if pct else ''
            return f"{m:.2f}{tag}+/-{s:.2f}"
        print(f"{'collisions (LoS)':<30}{_ms(base,'mean_collisions'):>18}"
              f"{_ms(f,'mean_collisions'):>18}")
        print(f"{'mean success rate':<30}{_ms(base,'mean_success_rate',100,True):>18}"
              f"{_ms(f,'mean_success_rate',100,True):>18}")
        print(f"{'corridor conformance':<30}{_ms(base,'mean_conformance_pct',1,True):>18}"
              f"{_ms(f,'mean_conformance_pct',1,True):>18}")
        print("-" * 66)
    print(f"{'metric':<30}{'baseline':>16}{'filtered':>16}")
    print(f"{'collisions (LoS)':<30}{base['mean_collisions']:>16.3f}{f['mean_collisions']:>16.3f}")
    print(f"{'  collisions cross-check':<30}{base['mean_los_pairsteps']:>16.3f}{f['mean_los_pairsteps']:>16.3f}")
    print(f"{'mean success rate':<30}{base['mean_success_rate']*100:>15.2f}%"
          f"{f['mean_success_rate']*100:>15.2f}%")
    print(f"{'corridor conformance':<30}{base['mean_conformance_pct']:>15.2f}%"
          f"{f['mean_conformance_pct']:>15.2f}%")
    print(f"{'collision-free episodes':<30}{base['collision_free_episode_pct']:>15.1f}%"
          f"{f['collision_free_episode_pct']:>15.1f}%")
    print(f"{'min separation (km)':<30}{base['min_closest_approach_km']:>16.3f}"
          f"{f['min_closest_approach_km']:>16.3f}")
    print(f"{'max intrusion depth (km)':<30}{base['mean_max_intrusion_km']:>16.3f}"
          f"{f['mean_max_intrusion_km']:>16.3f}")
    print(f"{'near-miss rate %':<30}{base['mean_nearmiss_rate_pct']:>16.2f}"
          f"{f['mean_nearmiss_rate_pct']:>16.2f}")
    print(f"{'throughput (agents/min)':<30}{base['mean_throughput']:>16.3f}"
          f"{f['mean_throughput']:>16.3f}")
    print(f"{'completion time (s, reached)':<30}{base['mean_completion_time_s']:>16.2f}"
          f"{f['mean_completion_time_s']:>16.2f}")
    _print_corridor_split(base, f)
    dc = f['mean_collisions'] - base['mean_collisions']
    ds = (f['mean_success_rate'] - base['mean_success_rate']) * 100
    dconf = f['mean_conformance_pct'] - base['mean_conformance_pct']
    print(f"\nchange: collisions {dc:+.3f}  ({100*dc/max(base['mean_collisions'],1e-9):+.1f}%),  "
          f"success {ds:+.2f} pts,  conformance {dconf:+.2f} pts")
    print("\nFilter behaviour:")
    print(f"  no-safe-action rate    : {f['no_safe_action_rate']*100:.2f}%  "
          f"(of {f['agent_steps']} agent-steps)")
    print(f"  avg primitives blocked : {f['avg_primitives_blocked']:.2f} / 25 per step")
    print("\n" + "-" * 62)
    if dc < 0 and ds >= -2.0:
        print("VERDICT: collisions DOWN, success held — filter is helping.")
    elif dc < 0 and ds < -2.0:
        print("VERDICT: collisions down but success dropped — filter too aggressive.")
    else:
        print("VERDICT: collisions NOT reduced — check the no-safe-action rate: "
              "high => primitives too coarse (escalate); low => barrier/tuning.")
    if dconf < -3.0:
        print(f"NOTE: corridor conformance fell {dconf:.1f} pts — the filter is pushing "
              "aircraft OUT of their corridors to avoid conflicts. A corridor-bounds "
              "term in the filter would address this.")


if __name__ == "__main__":
    main(sys.argv[1:])
