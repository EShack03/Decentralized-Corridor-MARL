"""
Ego-frame Control-Barrier-Function (CBF) safety filter.

Runs entirely on the in-process graph data the policy already receives
(node_obs / adj / obs / agent_id), so it needs no access to world.agents and
works unchanged under subprocess-vectorised training. Validated to reproduce the
world-frame geometry exactly (see cbf_egoframe_validate.py). One-step lookahead
(horizon = 1): for every motion primitive it rolls the ego one step, rebuilds the
ego's candidate graph, and keeps the primitive iff the learned barrier still
certifies the next state (h_next - h_now + alpha*h_now >= -eps) AND the predicted
nearest-neighbour distance stays >= the separation minimum. If no primitive
passes, the least-unsafe fallback keeps the single primitive that maximises the
predicted nearest-neighbour distance.

Node-feature layout (per ego graph): node_obs[:, :, 0:2] = relative velocity,
node_obs[:, :, 2:4] = relative position; obs[:, 2] = ego speed; agent_id = the
ego's own node index (the identifier the policy GNN itself uses).
"""

import numpy as np
import torch
from typing import Tuple


def build_primitive_table(num_accel_options: int, num_angrate_options: int,
                          accel_min: float, accel_max: float,
                          angular_rate_max: float) -> np.ndarray:
    """(K, 2) table of motion primitives [d_theta, d_v], indexed to match the
    environment's decode_action_index (k // num_accel selects the angular-rate
    row, k % num_accel selects the acceleration)."""
    accel_options = np.linspace(accel_min, accel_max, num_accel_options)
    angrate_options = np.linspace(-angular_rate_max, angular_rate_max,
                                  num_angrate_options)
    n_actions = num_accel_options * num_angrate_options
    table = np.zeros((n_actions, 2), dtype=np.float32)
    for k in range(n_actions):
        table[k, 0] = angrate_options[k // num_accel_options]
        table[k, 1] = accel_options[k % num_accel_options]
    return table


def ego_frame_next_dists(rel_pos, rel_vel, ego_speed, dtheta, dv, dt,
                         v_min, v_max):
    """Single-ego, single-primitive next-step neighbour distances (constant-
    velocity neighbours). Kept for validation/reference; the vectorised path
    lives in EgoCBFFilter.masks."""
    v = float(ego_speed)
    old_ego_vel = np.array([v, 0.0], dtype=np.float64)
    v_new = float(np.clip(v + dv * dt, v_min, v_max))
    ang = dtheta * dt
    new_ego_vel = np.array([v_new * np.cos(ang), v_new * np.sin(ang)],
                           dtype=np.float64)
    nbr_abs_vel = rel_vel.astype(np.float64) + old_ego_vel[None, :]
    new_rel_pos = rel_pos.astype(np.float64) + (nbr_abs_vel - new_ego_vel[None, :]) * dt
    return np.linalg.norm(new_rel_pos, axis=1)


class EgoCBFFilter:
    def __init__(self, barrier, prim_table, sep, dt, v_min, v_max,
                 alpha=1.0, eps=0.0, num_agents=None,
                 neighbour_model="static", fallback="least_unsafe",
                 device=None):
        self.barrier = barrier
        self.prim_table = prim_table
        self.K = prim_table.shape[0]
        self.sep = sep
        self.dt = dt
        self.v_min = v_min
        self.v_max = v_max
        self.alpha = alpha
        self.eps = eps
        self.num_agents = num_agents
        assert neighbour_model in ("static", "constant_velocity")
        self.neighbour_model = neighbour_model
        self.fallback = fallback
        self.device = device if device is not None else torch.device("cpu")

    @torch.no_grad()
    def masks(self, node_obs, adj, obs, agent_id):
        """
        node_obs : (B, N, F)   graph node features per ego (B = threads*agents)
        adj      : (B, N, N)   distance-magnitude matrix per ego
        obs      : (B, obs_dim) per-ego observation; obs[:, 2] = ego speed
        agent_id : (B, 1) or (B,) — the ego's node index in its own graph. This
                   is the canonical identifier the policy's GNN itself uses
                   (gnn_new.py gathers the ego node via agent_id), so we use it
                   directly rather than inferring the ego from a zero-rel_pos row.

        Returns
        -------
        avail  : (B, K) float32 0/1 mask (1 = primitive allowed)
        no_safe: (B,) bool — True where no primitive passed (fallback applied)
        """
        node_obs = np.asarray(node_obs, dtype=np.float32)
        adj = np.asarray(adj, dtype=np.float32)
        obs = np.asarray(obs, dtype=np.float32)
        ego_idx = np.asarray(agent_id).reshape(-1).astype(np.int64)
        B, N, _ = node_obs.shape
        na = self.num_agents
        K = self.K
        dt = self.dt

        ego_speed = obs[:, 2]                                # (B,)

        rel_vel = node_obs[:, :na, 0:2]                      # (B, na, 2)
        rel_pos = node_obs[:, :na, 2:4]                      # (B, na, 2)

        dth = self.prim_table[:, 0].astype(np.float32)       # (K,)
        dv = self.prim_table[:, 1].astype(np.float32)        # (K,)

        # ego kinematics over the lookahead, for every primitive
        v = ego_speed[:, None]                               # (B, 1)
        old_ego_vel = np.stack([v[:, 0], np.zeros(B, np.float32)], axis=1)  # (B, 2)
        v_new = np.clip(v + dv[None, :] * dt, self.v_min, self.v_max)       # (B, K)
        ang = dth[None, :] * dt                              # (1, K)
        ang = np.broadcast_to(ang, (B, K))                   # (B, K)
        new_ego_vel = np.stack([v_new * np.cos(ang), v_new * np.sin(ang)],
                               axis=-1)                      # (B, K, 2)

        # neighbour velocity model over the lookahead
        if self.neighbour_model == "static":
            nbr_abs_vel = np.zeros((B, na, 2), dtype=np.float32)            # (B, na, 2)
        else:
            nbr_abs_vel = rel_vel + old_ego_vel[:, None, :]                 # (B, na, 2)

        # predicted next-step relative positions and neighbour distances
        disp = (nbr_abs_vel[:, None, :, :] - new_ego_vel[:, :, None, :]) * dt  # (B, K, na, 2)
        new_rel_pos = rel_pos[:, None, :, :] + disp                            # (B, K, na, 2)
        new_dist = np.linalg.norm(new_rel_pos, axis=-1)                        # (B, K, na)

        # ignore the ego's own node and any zero (disconnected) entry
        col = np.arange(na)[None, :]                         # (1, na)
        is_self = col == ego_idx[:, None]                    # (B, na)
        big = np.broadcast_to(is_self[:, None, :], (B, K, na))
        new_dist = np.where(big, np.inf, new_dist)
        new_dist = np.where(new_dist == 0.0, np.inf, new_dist)
        min_dist = new_dist.min(axis=-1)                     # (B, K)

        # candidate next-step graph: copy each ego graph K times and overwrite the
        # ego row/column of the adjacency with the predicted next-step distances
        cand_adj = np.repeat(adj[:, None, :, :], K, axis=1)  # (B, K, N, N)
        bi = np.arange(B)[:, None]                           # (B, 1)
        ki = np.arange(K)[None, :]                           # (1, K)
        for j in range(na):
            cand_adj[bi, ki, ego_idx[:, None], j] = new_dist[:, :, j]
            cand_adj[bi, ki, j, ego_idx[:, None]] = new_dist[:, :, j]
        cand_adj[bi, ki, ego_idx[:, None], ego_idx[:, None]] = 0.0

        cand_node = np.repeat(node_obs[:, None, :, :], K, axis=1)  # (B, K, N, F)

        flat_node = torch.as_tensor(cand_node.reshape(B * K, N, -1),
                                    dtype=torch.float32, device=self.device)
        flat_adj = torch.as_tensor(cand_adj.reshape(B * K, N, N),
                                   dtype=torch.float32, device=self.device)
        h_next = self.barrier(flat_node, flat_adj).cpu().numpy()
        h_next = h_next.reshape(B, K, N)
        h_next_ego = h_next[bi, ki, ego_idx[:, None]]        # (B, K)

        base_node = torch.as_tensor(node_obs, dtype=torch.float32,
                                    device=self.device)
        base_adj = torch.as_tensor(adj, dtype=torch.float32, device=self.device)
        h_now = self.barrier(base_node, base_adj).cpu().numpy()
        h_now_ego = h_now[np.arange(B), ego_idx][:, None]    # (B, 1)

        # discrete barrier condition + hard separation floor
        keep = (h_next_ego - h_now_ego + self.alpha * h_now_ego) >= -self.eps  # (B, K)
        keep &= (min_dist >= self.sep)

        avail = keep.astype(np.float32)
        no_safe = ~keep.any(axis=1)                          # (B,)

        if no_safe.any():
            if self.fallback == "least_unsafe":
                best = np.argmax(min_dist, axis=1)           # (B,)
                avail[no_safe] = 0.0
                avail[no_safe, best[no_safe]] = 1.0
            else:
                avail[no_safe] = 1.0

        return avail, no_safe
