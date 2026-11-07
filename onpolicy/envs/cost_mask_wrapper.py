# ============================================================================
# cost_mask_wrapper.py - constrained-training wrapper around one GraphMPE env
# ----------------------------------------------------------------------------
# [ES] added (constrained training, Nov 2026). New file; nothing in the original
# environment changes. The policy sees exactly the same observations, rewards
# and dynamics as before; this wrapper only ADDS entries to each agent's info
# dict, which the training runner (graph_mpe_runner_lag.py) reads.
#
# Runs inside each rollout worker process, because the safety filter needs the
# true world state (positions, headings, speeds), which only the worker has.
#
# Added to info[i] after every env.step():
#   lag_cost  [separation_cost, corridor_cost] for agent i on this step
#             separation_cost = (other active aircraft closer than 152 m)
#                               + w_near * (other active aircraft 152-304 m away)
#             corridor_cost   = 1 while agent i is off its corridor: it left the
#                               side of its current corridor between the entrance
#                               and the exit (|lateral| > half width + 50 m), and
#                               it stays 1 until the aircraft is back inside.
#                               The queue before the entrance (s < 0), the
#                               junction gap and the run to the goal after a
#                               proper exit (s > L) never count.
#             Aircraft parked at their goal cost 0.
#   lag_mask    (25,) training-filter mask for the NEXT decision (1 = allowed),
#               or absent when the filter is off (with --lag_filter False it is
#               still present during the first --lag_warmup episodes)
#   lag_nosafe  True when no primitive passed the training filter (the least-
#               unsafe fallback was applied)
#   lag_safe1   (25,) 1-second filter mask for the mask-consistency loss, or
#               absent when that loss is off
#
# The masks describe the state AFTER the step, i.e. the state the policy acts
# on next. When the worker auto-resets the env at the end of an episode it
# calls reset() before sending the infos back, so reset() rewrites the masks in
# the same info dicts to describe the reset state (the costs stay those of the
# step that ended the episode).
#
# Training filter = the plain filter (S2_ip): constant-velocity neighbours,
# fixed look-ahead, alpha = 1, parked aircraft ignored, no caution dial, no gap
# guard, least-unsafe fallback. The barrier network is the Stage-1 barrier.
# ============================================================================
import numpy as np
import torch

# air_taxi constants, the same values eval_mpe.py uses for the eval filter
_KTS = 0.000514444
SEP_KM = 500 * 0.0003048            # 152.4 m loss-of-separation distance
V_MIN, V_MAX = 60 * _KTS, 175 * _KTS
ACCEL_MIN, ACCEL_MAX, ANG_RATE_MAX = -0.001, 0.002, 0.075
N_ACCEL, N_ANGRATE = 5, 5
DT = 1.0
MAX_EDGE = 2 * 1.60934
CORRIDOR_EPS = 0.05                 # km, the scenarios' own in-tube tolerance


class CostMaskEnv:
    """Wraps one MultiAgentGraphEnv. Every attribute not defined here is the
    wrapped env's (spaces, world, seed, close, ...)."""

    def __init__(self, env, args):
        self.env = env
        self.w_near = float(getattr(args, "lag_near_weight", 0.25))
        self._warned = False
        self._last_info = None
        self.filter_h = None
        self.filter_1 = None
        self.filter_on = bool(getattr(args, "lag_filter", True))
        # with the filter off in training, the first --lag_warmup episodes still
        # run with it, so the warm-up measures the budget the policy must reach
        # on its own (the original policy's cost WITH the filter)
        self.warm_episodes = 0 if self.filter_on else int(getattr(args, "lag_warmup", 0))
        self.episode = -1                    # becomes 0 at the first reset
        need_h = self.filter_on or self.warm_episodes > 0
        need_1 = float(getattr(args, "lag_mask_coef", 0.0)) > 0.0
        if need_h or need_1:
            torch.set_num_threads(1)
            from onpolicy.algorithms.utils.cbf_gnn import CBFGNN
            from onpolicy.algorithms.utils.cbf_filter import (
                CBFSafetyFilter, build_primitive_table)
            dev = torch.device("cpu")
            prim = build_primitive_table(N_ACCEL, N_ANGRATE, ACCEL_MIN, ACCEL_MAX,
                                         ANG_RATE_MAX)
            barrier = CBFGNN(node_obs_dim=7, edge_dim=1,
                             hidden_size=getattr(args, "cbf_hidden", 64),
                             num_layers=getattr(args, "cbf_layers", 2),
                             max_edge_dist=MAX_EDGE).to(dev)
            path = getattr(args, "lag_barrier", "cbf_stage1_barrier.pt")
            barrier.load_state_dict(torch.load(path, map_location=dev))
            barrier.eval()

            def make(h):
                return CBFSafetyFilter(
                    barrier=barrier, prim_table=prim, sep=SEP_KM, dt=DT,
                    v_min=V_MIN, v_max=V_MAX, alpha=1.0, eps=0.0, horizon=h,
                    fallback="least_unsafe", num_agents=args.num_agents,
                    neighbour_model="constant_velocity", device=dev,
                    ignore_parked=True)
            H = int(getattr(args, "lag_filter_horizon", 3))
            if need_h:
                self.filter_h = make(H)
            if need_1:
                self.filter_1 = self.filter_h if (need_h and H == 1) else make(1)
        self._off = None
        # graph padding for mixed layouts: the merge layouts have no landmark
        # nodes (graph = aircraft only), single/sequential have aircraft + goals,
        # but one training buffer needs one graph size. Extra nodes are all-zero
        # with no edges (adj 0 = no edge in the GNN), so the actor's output for
        # the ego node is unchanged; only the critics' global mean sees them.
        self.pad = int(getattr(args, "lag_pad_nodes", 0) or 0)
        if self.pad < 0:                     # automatic: pad only if layouts mix
            lay = [s for s in (getattr(args, "lag_scenarios", "") or "").split(",") if s]
            mixed = any(s in ("single", "sequential") for s in lay) and \
                any(s in ("merge", "mergeseq") for s in lay)
            self.pad = 2 * int(args.num_agents) if mixed else 0
        if self.pad:
            from gymnasium import spaces as _sp
            nobs = self.env.node_observation_space[0].shape
            if self.pad < nobs[0]:
                raise ValueError(f"--lag_pad_nodes {self.pad} < graph size {nobs[0]}")
            self.node_observation_space = [
                _sp.Box(low=-np.inf, high=np.inf, shape=(self.pad, nobs[1]), dtype=np.float32)
                for _ in self.env.node_observation_space]
            self.adj_observation_space = [
                _sp.Box(low=-np.inf, high=np.inf, shape=(self.pad, self.pad), dtype=np.float32)
                for _ in self.env.adj_observation_space]

    def _pad(self, node_obs_n, adj_n):
        if not self.pad:
            return node_obs_n, adj_n
        P = self.pad
        no2, ad2 = [], []
        for no, ad in zip(node_obs_n, adj_n):
            no = np.asarray(no); ad = np.asarray(ad)
            n = no.shape[0]
            if n == P:
                no2.append(no); ad2.append(ad); continue
            x = np.zeros((P, no.shape[1]), dtype=no.dtype); x[:n] = no
            y = np.zeros((P, P), dtype=ad.dtype); y[:n, :n] = ad
            no2.append(x); ad2.append(y)
        return no2, ad2

    def __getattr__(self, name):
        return getattr(self.env, name)

    # ------------------------------------------------------------------ helpers
    def _scen(self):
        cb = getattr(self.env, "info_callback", None)
        return getattr(cb, "__self__", None)

    def _agents(self):
        return self.env.world.agents

    def _beside_inside(self, agents):
        """Per agent (beside, inside) its current corridor: beside = between the
        entrance and the exit (0 <= s <= L) but |lateral| > half width + 50 m;
        inside = 0 <= s <= L and |lateral| <= half width + 50 m."""
        n = len(agents)
        beside, inside = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
        scen = self._scen()
        if scen is None or not hasattr(scen, "_tube_coords"):
            return beside, inside
        multi = getattr(scen, "current_tube", None) is not None
        try:
            for i, a in enumerate(agents):
                if multi:
                    s, y, L, hw = scen._tube_coords(self.env.world, a.state.p_pos,
                                                    int(scen.current_tube[a.id]))
                else:
                    s, y, L, hw = scen._tube_coords(self.env.world, a.state.p_pos)
                along = 0.0 <= s <= L
                beside[i] = along and abs(y) > hw + CORRIDOR_EPS
                inside[i] = along and abs(y) <= hw + CORRIDOR_EPS
        except Exception as exc:                     # never break a rollout
            if not self._warned:
                print(f"[lag] corridor cost unavailable ({exc}); using 0")
                self._warned = True
            beside[:] = False
            inside[:] = False
        return beside, inside

    def _masks(self, node_obs_n, adj_n):
        """(mask_h, nosafe, mask_1) for the current state, each None if off."""
        if self.filter_h is None and self.filter_1 is None:
            return None, None, None
        ags = self._agents()
        ws = np.array([[a.state.p_pos[0], a.state.p_pos[1], float(a.state.theta),
                        float(a.state.speed)] for a in ags], dtype=np.float32)[None]
        active = np.array([not bool(a.status) for a in ags], dtype=bool)[None]
        no = np.asarray(node_obs_n, dtype=np.float32)[None]
        ad = np.asarray(adj_n, dtype=np.float32)[None]
        mh = ns = m1 = None
        use_h = self.filter_h is not None and (self.filter_on or
                                               self.episode < self.warm_episodes)
        if use_h:
            mh, ns = self.filter_h.masks_for_threads(no, ad, ws, agent_active=active)
            mh, ns = mh[0], ns[0]
        if self.filter_1 is not None:
            if self.filter_1 is self.filter_h and mh is not None:
                m1 = mh
            else:
                m1, _ = self.filter_1.masks_for_threads(no, ad, ws, agent_active=active)
                m1 = m1[0]
        return mh, ns, m1

    @staticmethod
    def _put_masks(info_n, mh, ns, m1):
        for i, info in enumerate(info_n):
            if mh is not None:
                info["lag_mask"] = mh[i]
                info["lag_nosafe"] = bool(ns[i])
            if m1 is not None:
                info["lag_safe1"] = m1[i]

    # ------------------------------------------------------------------ env API
    def step(self, actions):
        obs_n, agent_id_n, node_obs_n, adj_n, reward_n, done_n, info_n = \
            self.env.step(actions)
        ags = self._agents()
        n = len(ags)
        if self._off is None or len(self._off) != n:
            self._off = np.zeros(n, dtype=bool)
        active = np.array([not bool(a.status) for a in ags], dtype=bool)
        P = np.array([a.state.p_pos for a in ags], dtype=np.float64)
        D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
        np.fill_diagonal(D, np.inf)
        D[:, ~active] = np.inf                       # parked aircraft are gone
        n_coll = (D < SEP_KM).sum(axis=1)
        n_near = ((D >= SEP_KM) & (D < 2 * SEP_KM)).sum(axis=1)
        sep_cost = (n_coll + self.w_near * n_near) * active

        beside, inside = self._beside_inside(ags)
        self._off = (self._off | beside) & active & ~inside   # back inside clears it
        corr_cost = self._off.astype(np.float64)

        for i, info in enumerate(info_n):
            info["lag_cost"] = np.array([sep_cost[i], corr_cost[i]], dtype=np.float32)
        mh, ns, m1 = self._masks(node_obs_n, adj_n)
        self._put_masks(info_n, mh, ns, m1)
        self._last_info = info_n
        node_obs_n, adj_n = self._pad(node_obs_n, adj_n)
        return obs_n, agent_id_n, node_obs_n, adj_n, reward_n, done_n, info_n

    def reset(self, *args, **kwargs):
        obs_n, agent_id_n, node_obs_n, adj_n = self.env.reset(*args, **kwargs)
        self.episode += 1
        self._off = np.zeros(len(self._agents()), dtype=bool)
        if self._last_info is not None:              # auto-reset inside a step
            mh, ns, m1 = self._masks(node_obs_n, adj_n)
            self._put_masks(self._last_info, mh, ns, m1)
        self._last_info = None
        node_obs_n, adj_n = self._pad(node_obs_n, adj_n)
        return obs_n, agent_id_n, node_obs_n, adj_n
