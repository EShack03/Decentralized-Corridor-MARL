"""
Stage 1 (real) — train the CBFGNN barrier network on rollouts from the frozen
existing policy, then plot h vs inter-agent distance (the Stage-1 verification gate).

This deliberately reuses the SAME machinery as eval_mpe.py to build the env and
load the trained policy, so the barrier sees exactly the graphs the policy sees.
No part of the policy is modified or trained here — only the barrier network.

USAGE (from repo root, venv active), mirroring how you launch eval:

    python onpolicy/scripts/train_cbf_stage1.py \
        --model_dir "model_weights/tube/rot_inv/airtaxi/try/three/test2026/5ag/low_width" \
        --env_name GraphMPE --algorithm_name rmappo \
        --scenario_name three_phase_graph_sequential \
        --dynamics_type air_taxi --num_agents 5 --num_landmarks 5 \
        --world_size 5 --episode_length 120 \
        --cbf_rollout_episodes 40 --cbf_epochs 300

It writes:
    cbf_stage1_barrier.pt        (trained barrier weights)
    cbf_stage1_boundary.png      (the h-vs-distance gate plot)

The last two args (--cbf_rollout_episodes, --cbf_epochs) are added below; every
other arg is the standard eval arg set, so pass the same ones you use for eval.
"""

import argparse
import os
import sys
from pathlib import Path

# Put the repo root on the path BEFORE any onpolicy import, exactly as
# eval_mpe.py does — this is why `python onpolicy/scripts/...` works despite
# onpolicy not being pip-installed.
sys.path.append(os.path.abspath(os.getcwd()))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the project's config + env + arg-modification exactly as eval does.
from onpolicy.config import get_config
from onpolicy.scripts.eval_mpe import make_render_env, parse_args, modify_args

from onpolicy.algorithms.utils.cbf_gnn import CBFGNN, safe_unsafe_masks, cbf_loss


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def nearest_neighbour_dist(adj_b_nn: np.ndarray,
                           ego_idx: np.ndarray,
                           num_agents: int) -> np.ndarray:
    """
    Per-sample nearest-OTHER-AGENT distance.

    Each sample is one agent's view at one step. `adj_b_nn` is the (shared,
    global) distance-magnitude matrix for that sample, shape
    (batch, num_nodes, num_nodes); `ego_idx[b]` is the row/column of the agent
    whose safety this sample is about. The agents occupy the first `num_agents`
    nodes (agents come before landmarks in the entity ordering).

    For each sample we read the ego agent's row, restrict to the agent columns,
    drop the self entry and any zero (masked / disconnected) entries, and take
    the minimum. Returns (batch,).
    """
    b = adj_b_nn.shape[0]
    row = adj_b_nn[np.arange(b), ego_idx, :num_agents].copy()   # (b, num_agents)
    row[np.arange(b), ego_idx] = np.inf                         # ignore self
    row[row == 0.0] = np.inf                                    # ignore disconnected
    return row.min(axis=-1)                                     # (b,)


def collect_rollouts(runner, episodes: int):
    """
    Run the frozen policy for `episodes` episodes and collect, at every step:
        node_obs(t), adj(t), node_obs(t+1), adj(t+1)
    Returns tensors ready for the barrier network.
    """
    envs = runner.eval_envs
    n_threads = runner.n_eval_rollout_threads
    num_agents = runner.num_agents

    node_obs_t, adj_t, node_obs_tp1, adj_tp1, ego_idx = [], [], [], [], []

    runner.trainer.prep_rollout()

    eps_done = 0
    while eps_done < episodes:
        obs, agent_id, node_obs, adj = envs.reset()
        rnn_states = np.zeros((n_threads, *runner.buffer.rnn_states.shape[2:]),
                              dtype=np.float32)
        masks = np.ones((n_threads, num_agents, 1), dtype=np.float32)

        for _ in range(runner.episode_length):
            prev_node_obs, prev_adj = node_obs.copy(), adj.copy()

            action, rnn_states = runner.trainer.policy.act(
                np.concatenate(obs),
                np.concatenate(node_obs),
                np.concatenate(adj),
                np.concatenate(agent_id),
                np.concatenate(rnn_states),
                np.concatenate(masks),
                deterministic=True,
            )
            action = np.array(np.split(action.detach().cpu().numpy(), n_threads))
            rnn_states = np.array(np.split(rnn_states.detach().cpu().numpy(), n_threads))

            # one-hot the discrete action for the env step (matches runner.eval)
            aspace = envs.action_space[0]
            if aspace.__class__.__name__ == "Discrete":
                actions_env = np.squeeze(np.eye(aspace.n)[action], 2)
            elif aspace.__class__.__name__ == "MultiDiscrete":
                parts = []
                for i in range(aspace.shape):
                    parts.append(np.eye(aspace.high[i] + 1)[action[:, :, i]])
                actions_env = np.concatenate(parts, axis=2)
            else:
                raise NotImplementedError(aspace.__class__.__name__)

            # The single-thread GraphDummyVecEnv returns an extra trailing
            # `reset_count` (8 values); the Subproc form returns 7. Unpack the
            # first 7 either way.
            step_out = envs.step(actions_env)
            obs, agent_id, node_obs, adj, rewards, dones, infos = step_out[:7]

            # NOTE: GraphDummyVecEnv auto-resets on done inside step_wait, so
            # when the episode ends the returned node_obs/adj are already the
            # NEXT episode's reset state. That would make the final (t -> t+1)
            # transition cross an episode boundary and corrupt h_dot, so we
            # skip storing it and end the episode instead.
            if np.all(dones):
                break

            # Store one SAMPLE per (thread, agent). Each agent has its own
            # node_obs view (features relative to that ego agent) and shares the
            # global distance matrix. The ego agent occupies node index `a`.
            #   prev_node_obs[th] : (n_agents, n_nodes, feat)
            #   prev_adj[th]      : (n_agents, n_nodes, n_nodes)
            for th in range(n_threads):
                for a in range(num_agents):
                    node_obs_t.append(prev_node_obs[th][a])
                    adj_t.append(prev_adj[th][a])
                    node_obs_tp1.append(node_obs[th][a])
                    adj_tp1.append(adj[th][a])
                    ego_idx.append(a)

        eps_done += n_threads
        print(f"  collected {eps_done}/{episodes} episodes", flush=True)

    to_t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)
    return (to_t(node_obs_t), to_t(adj_t), to_t(node_obs_tp1), to_t(adj_tp1),
            np.asarray(ego_idx, dtype=np.int64))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main(argv):
    parser = get_config()
    # Add the Stage-1-specific args to the parser FIRST. (eval's parse_args
    # returns a parsed Namespace, not the parser, so our args must go on the
    # parser before parse_args consumes it.)
    parser.add_argument("--cbf_rollout_episodes", type=int, default=40,
                        help="episodes of frozen-policy rollout to train the barrier on")
    parser.add_argument("--cbf_epochs", type=int, default=300)
    parser.add_argument("--cbf_hidden", type=int, default=64)
    parser.add_argument("--cbf_layers", type=int, default=2)
    parser.add_argument("--cbf_alpha", type=float, default=1.0)
    parser.add_argument("--cbf_lr", type=float, default=1e-3)
    parser.add_argument("--cbf_margin", type=float, default=2.0,
                        help="safe band = margin * separation_distance")
    # parse_args adds eval's own args (scenario_name, num_agents, ...) and
    # returns the fully parsed Namespace.
    all_args = parse_args(argv, parser)

    # Fill missing args from the model's saved config, exactly like eval.
    all_args = modify_args(all_args.model_dir, all_args)

    # Force render mode so the runner skips training-dir creation and the
    # critic load (base_runner guards both behind `if not self.use_render:`).
    # We only need the actor (the frozen policy) and the eval env here.
    all_args.use_render = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)

    # --- Build env + runner (loads the frozen policy) via the eval path ---
    envs = make_render_env(all_args)
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": Path("."),
    }
    from onpolicy.runner.shared.graph_mpe_runner import GMPERunner as Runner
    runner = Runner(config)

    # separation distance + sensing radius come straight from the scenario config
    # (COLLISION_DISTANCE = 0.1524 km, COORDINATION_RANGE = 3.219 km for air_taxi)
    from multiagent.config import AirTaxiConfig
    sep = float(AirTaxiConfig.COLLISION_DISTANCE)
    max_edge = float(AirTaxiConfig.COORDINATION_RANGE)
    print(f"separation_distance = {sep:.4f} km   sensing/max_edge = {max_edge:.4f} km")

    # --- Collect rollouts from the frozen policy ---
    print(f"Collecting {all_args.cbf_rollout_episodes} rollout episodes ...")
    no_t, adj_t, no_tp1, adj_tp1, ego = collect_rollouts(
        runner, all_args.cbf_rollout_episodes)
    node_obs_dim = no_t.shape[-1]
    print(f"collected samples: {no_t.shape[0]}   node_obs_dim = {node_obs_dim}")

    # nearest-OTHER-AGENT distance per sample (for labels), from the ego row of adj
    nn_dist = torch.as_tensor(
        nearest_neighbour_dist(adj_t.numpy(), ego, num_agents), dtype=torch.float32)
    ego_t = torch.as_tensor(ego, dtype=torch.long)

    # --- Build + train the barrier network ---
    net = CBFGNN(node_obs_dim=node_obs_dim, edge_dim=1,
                 hidden_size=all_args.cbf_hidden, num_layers=all_args.cbf_layers,
                 max_edge_dist=max_edge).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=all_args.cbf_lr)

    no_t, adj_t = no_t.to(device), adj_t.to(device)
    no_tp1, adj_tp1 = no_tp1.to(device), adj_tp1.to(device)
    nn_dist = nn_dist.to(device)
    ego_t = ego_t.to(device)
    batch_ar = torch.arange(no_t.shape[0], device=device)

    safe_mask, unsafe_mask = safe_unsafe_masks(nn_dist, sep, margin=all_args.cbf_margin)
    print(f"labelled: {int(safe_mask.sum())} safe / {int(unsafe_mask.sum())} unsafe "
          f"agent-steps (of {nn_dist.numel()})")

    print("Training barrier network ...")
    for epoch in range(all_args.cbf_epochs):
        opt.zero_grad()
        # h for each sample is read at that sample's ego node.
        h = net(no_t, adj_t)[batch_ar, ego_t]
        h_next = net(no_tp1, adj_tp1)[batch_ar, ego_t]
        loss, comp = cbf_loss(h, h_next, safe_mask, unsafe_mask, alpha=all_args.cbf_alpha)
        loss.backward()
        opt.step()
        if epoch % 25 == 0 or epoch == all_args.cbf_epochs - 1:
            print(f"  epoch {epoch:4d}  {comp}", flush=True)

    torch.save(net.state_dict(), "cbf_stage1_barrier.pt")
    print("saved cbf_stage1_barrier.pt")

    # --- The Stage-1 GATE: plot h vs nearest-neighbour distance ---
    net.eval()
    with torch.no_grad():
        h = net(no_t, adj_t)[batch_ar, ego_t].cpu().numpy().reshape(-1)
    d = nn_dist.cpu().numpy().reshape(-1)
    finite = np.isfinite(d)
    d, h = d[finite], h[finite]

    plt.figure(figsize=(8, 5))
    plt.scatter(d, h, s=4, alpha=0.25, color="#3B82F6")
    plt.axvline(sep, color="#EF4444", ls="--", label=f"separation = {sep:.3f} km")
    plt.axhline(0.0, color="#111827", lw=1)
    plt.xlabel("distance to nearest other aircraft (km)")
    plt.ylabel("barrier value  h")
    plt.title("Stage-1 gate: h should be < 0 left of the line, > 0 right of it")
    plt.legend()
    plt.tight_layout()
    plt.savefig("cbf_stage1_boundary.png", dpi=150)
    print("saved cbf_stage1_boundary.png")

    # quick numeric gate summary
    left = h[d < sep]
    right = h[d > all_args.cbf_margin * sep]
    if left.size and right.size:
        print(f"GATE:  unsafe side mean h = {left.mean():+.3f} (want <0), "
              f"frac correct = {(left < 0).mean():.2f}")
        print(f"       safe   side mean h = {right.mean():+.3f} (want >0), "
              f"frac correct = {(right >= 0).mean():.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
