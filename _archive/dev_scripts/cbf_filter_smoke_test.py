"""
Smoke test for the CBF safety filter (Stage 2), CPU + dummy data, no env needed.

Checks:
  1. build_primitive_table reproduces environment.decode_action_index exactly
     for the air_taxi 5x5 grid (25 primitives).
  2. step_air_taxi advances a known state the way the kinematics say it should.
  3. The filter returns a length-25 0/1 mask, and — with a barrier that simply
     reports "distance to nearest neighbour minus separation" — it blocks the
     primitives that drive the ego INTO a neighbour and keeps the ones that
     steer away. This exercises the whole masking path without the real barrier.

Run from the utils/ folder:
    python3 cbf_filter_smoke_test.py
"""

import numpy as np
import torch

from cbf_filter import (build_primitive_table, step_air_taxi, CBFSafetyFilter)


# air_taxi constants (from multiagent/config.AirTaxiConfig)
N_ACCEL = 5
N_ANGRATE = 5
ACCEL_MIN = -0.001
ACCEL_MAX = 0.002
ANG_RATE_MAX = 0.075
DT = 1.0
V_MIN = 60 * 0.514444 * 0.001
V_MAX = 175 * 0.514444 * 0.001
SEP = 0.1524


def decode_reference(action_index, n_accel, n_angrate, amin, amax, wmax):
    """Reproduce environment.decode_action_index (air_taxi) for cross-checking."""
    accel_options = np.linspace(amin, amax, n_accel)
    angle_rate_options = np.linspace(-wmax, wmax, n_angrate)
    angle_rate_index = action_index // n_accel
    accel_index = action_index % n_accel
    return np.array([angle_rate_options[angle_rate_index], accel_options[accel_index]])


class DistanceBarrier(torch.nn.Module):
    """
    A stand-in 'barrier' with no learned params: h(ego) = (nearest-neighbour
    distance in adj) - separation. Positive when safe, negative when too close —
    exactly the sign convention of the real CBFGNN, so the filter logic can be
    tested deterministically.
    """
    def __init__(self, num_agents, sep):
        super().__init__()
        self.num_agents = num_agents
        self.sep = sep

    def forward(self, node_obs, adj):
        # adj: (B, N, N). h per node = min over other agent-nodes of distance - sep
        B, N, _ = adj.shape
        na = self.num_agents
        block = adj[:, :na, :na].clone()
        eye = torch.eye(na, dtype=torch.bool)[None].expand(B, na, na)
        block[eye] = float("inf")
        block[block == 0] = float("inf")
        nn = block.min(dim=-1).values          # (B, na)
        h = nn - self.sep
        # pad non-agent nodes with a big positive value
        out = torch.full((B, N), 10.0)
        out[:, :na] = h
        return out


def main():
    # ---- 1. primitive table matches the decoder ----
    table = build_primitive_table(N_ACCEL, N_ANGRATE, ACCEL_MIN, ACCEL_MAX, ANG_RATE_MAX)
    assert table.shape == (25, 2), table.shape
    ok = True
    for k in range(25):
        ref = decode_reference(k, N_ACCEL, N_ANGRATE, ACCEL_MIN, ACCEL_MAX, ANG_RATE_MAX)
        if not np.allclose(table[k], ref, atol=1e-9):
            ok = False
            print(f"  primitive {k}: table={table[k]} ref={ref}")
    assert ok, "primitive table does not match decode_action_index"
    print(f"[1] primitive table matches decode_action_index for all 25 primitives")

    # ---- 2. kinematics sanity ----
    # heading 0, speed s: after dt with dtheta=0,dv=0 -> moves +x by s*dt
    s = V_MIN
    pos0 = np.array([0.0, 0.0], dtype=np.float32)
    p, th, v = step_air_taxi(pos0, 0.0, s, 0.0, 0.0, DT, V_MIN, V_MAX)
    assert np.allclose(p, [s * DT, 0.0], atol=1e-6), p
    assert abs(th) < 1e-9 and abs(v - s) < 1e-9
    print(f"[2] kinematics: straight step moves +x by speed*dt = {s*DT:.5f} km  OK")

    # ---- 3 & 4. filter blocks moves that close on a neighbour ----
    # The discriminating geometry depends on horizon: over H steps the ego travels
    # ~H*step_len, so the neighbour must sit further away for larger H. We build a
    # correctly-posed scenario per horizon and check the filter discriminates.
    num_agents = 2
    num_nodes = 3            # 2 agents + 1 landmark
    barrier = DistanceBarrier(num_agents, SEP)
    v_mid = 0.5 * (V_MIN + V_MAX)
    ego_pos = np.array([0.0, 0.0], dtype=np.float32)

    def run_case(horizon, tag):
        filt = CBFSafetyFilter(
            barrier=barrier, prim_table=table, sep=SEP, dt=DT,
            v_min=V_MIN, v_max=V_MAX, alpha=1.0, eps=0.0,
            horizon=horizon, num_agents=num_agents, device=torch.device("cpu"))

        # Roll the fastest and slowest straight primitives forward `horizon` steps
        # to find where each lands, then place the neighbour in the band between
        # them (+ sep) so the fast one breaches within the horizon and the slow one
        # does not — a genuine discrimination test at this horizon.
        pf = ego_pos.copy(); tf = 0.0; vf = v_mid
        ps = ego_pos.copy(); ts = 0.0; vs = v_mid
        for _ in range(horizon):
            pf, tf, vf = step_air_taxi(pf, tf, vf, 0.0, ACCEL_MAX, DT, V_MIN, V_MAX)
            ps, ts, vs = step_air_taxi(ps, ts, vs, 0.0, ACCEL_MIN, DT, V_MIN, V_MAX)
        nbr_x = 0.5 * ((ps[0] + SEP) + (pf[0] + SEP))
        nbr_pos = np.array([nbr_x, 0.0], dtype=np.float32)

        world_states = np.array([
            [ego_pos[0], ego_pos[1], 0.0, v_mid],
            [nbr_pos[0], nbr_pos[1], np.pi, V_MIN],
        ], dtype=np.float32)[None]

        node_obs = np.zeros((1, num_agents, num_nodes, 7), dtype=np.float32)
        pos_all = np.array([ego_pos, nbr_pos, [5.0, 5.0]], dtype=np.float32)
        d = np.linalg.norm(pos_all[:, None] - pos_all[None], axis=2).astype(np.float32)
        adj = np.repeat(d[None, None], num_agents, axis=1)

        available, no_safe = filt.masks_for_threads(node_obs, adj, world_states)
        ego_mask = available[0, 0]
        n_allowed = int(ego_mask.sum())
        straight_max = 2 * N_ACCEL + 4
        straight_min = 2 * N_ACCEL + 0
        print(f"[{tag}] horizon={horizon}: {n_allowed}/25 allowed, no_safe={no_safe[0,0]}  "
              f"| max-accel allowed={bool(ego_mask[straight_max])} (want F), "
              f"max-decel allowed={bool(ego_mask[straight_min])} (want T)")

        assert not no_safe[0, 0], f"horizon={horizon}: fell back to unfiltered (ill-posed)"
        assert ego_mask[straight_max] == 0.0, f"horizon={horizon}: failed to block closing move"
        assert ego_mask[straight_min] == 1.0, f"horizon={horizon}: wrongly blocked safe move"
        assert 1 <= n_allowed < 25, f"horizon={horizon}: expected partial mask, got {n_allowed}"

    run_case(1, "3")
    run_case(3, "4")

    print("\nSMOKE TEST PASSED — at horizon 1 and 3 the filter blocks the closing "
          "move, keeps the safe one, and returns a valid 0/1 mask of length 25.")


if __name__ == "__main__":
    main()
