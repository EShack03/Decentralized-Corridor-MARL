"""
CBF safety filters — two frames of the same safety layer, in one module.

This file holds BOTH control-barrier-function safety filters used in the
project; they implement the identical safety rule (keep a motion primitive iff
the learned barrier still certifies the next state AND the predicted
nearest-neighbour distance stays >= the separation minimum), differing only in
what state they read it from:

  * CBFSafetyFilter  (world frame) — used at EVALUATION. Reads the ego's absolute
    world state [x, y, theta, v] from the env and integrates the air_taxi
    kinematics forward with within-step sub-integration, over a configurable
    horizon. This is the reference implementation.

  * EgoCBFFilter     (ego frame)   — used during TRAINING. Reads only the
    in-process graph tensors the policy already receives (node_obs / adj / obs /
    agent_id), so it needs no access to world.agents and works under
    subprocess-vectorised training. One-step lookahead (horizon = 1).

The two were validated to produce bitwise-identical masks on shared inputs (max
geometric error ~9e-16 km), so a policy trained under EgoCBFFilter is filtered at
eval by exactly the same rule via CBFSafetyFilter. They were kept in separate
files during development and are consolidated here; the shared primitive table
(build_primitive_table) is defined once, below, and used by both.

================================ world-frame (eval) ===========================

CBF safety filter (Stage 2 of Option B).

Given the frozen barrier network (Stage 1) and the current graph, this module
decides — for each aircraft — which of its 25 discrete motion primitives are
*safe* to take, and returns a length-`n_actions` availability mask per agent in
exactly the 0/1 form the existing policy already consumes (`available_actions`:
a 0 blocks that primitive by driving its logit to -inf inside the masked
Categorical, see onpolicy/algorithms/utils/distributions.py).

How a primitive is judged (ego-only lookahead — the decentralised GCBF
assumption):
  1. Decode the primitive index k -> (dtheta, dv), exactly as
     environment.decode_action_index does for air_taxi.
  2. Integrate the ego aircraft ONE step with the air_taxi kinematics
     (multiagent/core.AirTaxiXYState.dstate). Neighbours are propagated according
     to `neighbour_model`: "static" holds them at their current position;
     "constant_velocity" advances them along their current heading at their
     current speed over the lookahead. This gives the ego's and neighbours'
     predicted next poses.
  3. Rebuild the ego's row/column of the distance matrix `adj` from those
     predicted poses (the separation signal the barrier keys on), and update the
     ego's ego-relative node features with the same rotation transform the
     scenario uses.
  4. Evaluate the barrier at the predicted next graph: h_next = barrier(...)[ego].
  5. Keep the primitive iff the discrete barrier condition holds:
         h_next - h_now + alpha * h_now >= -eps      (i.e. h_next >= (1-alpha)*h_now - eps)
     which is the one-step form of  h_dot + alpha*h >= 0.

If NO primitive passes for an agent, the mask would be all-zero, which the
Categorical cannot handle (it renormalises over available actions). In that case
we fall back to leaving that agent UNFILTERED (all primitives available) and flag
it, so the "no safe primitive available" rate can be logged — this rate is the
decision instrument for the escalation ladder (coarser vs finer vs continuous
action set). Blocking everything would freeze the aircraft and is never safer.

Nothing here trains or modifies the policy or the barrier.
"""

from typing import Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# air_taxi primitive table + one-step kinematic rollout (module-level, cached)
# ---------------------------------------------------------------------------

def build_primitive_table(num_accel_options: int,
                          num_angrate_options: int,
                          accel_min: float,
                          accel_max: float,
                          angular_rate_max: float) -> np.ndarray:
    """
    Return an (n_actions, 2) table mapping primitive index k -> (dtheta, dv),
    reproducing environment.decode_action_index for the air_taxi case exactly:

        angle_rate_index = k // num_accel_options
        accel_index      = k %  num_accel_options
        dtheta = linspace(-angular_rate_max, +angular_rate_max, num_angrate)[angle_rate_index]
        dv     = linspace(accel_min, accel_max, num_accel)[accel_index]
    """
    accel_options = np.linspace(accel_min, accel_max, num_accel_options)
    angrate_options = np.linspace(-angular_rate_max, angular_rate_max, num_angrate_options)

    n_actions = num_accel_options * num_angrate_options
    table = np.zeros((n_actions, 2), dtype=np.float32)
    for k in range(n_actions):
        angle_rate_index = k // num_accel_options
        accel_index = k % num_accel_options
        table[k, 0] = angrate_options[angle_rate_index]   # dtheta
        table[k, 1] = accel_options[accel_index]          # dv
    return table


def step_air_taxi(pos: np.ndarray, theta: float, speed: float,
                  dtheta: float, dv: float, dt: float,
                  v_min: float, v_max: float,
                  substeps: int = 20) -> Tuple[np.ndarray, float, float]:
    """
    One-step forward kinematics for the air_taxi model, matching
    AirTaxiXYState.dstate + update_state.

        xdot = v*cos(theta),  ydot = v*sin(theta),  thetadot = dtheta,  vdot = dv

    IMPORTANT: the simulator integrates the COUPLED ODE with RK45 over [0, dt],
    so heading and speed evolve DURING the interval and thereby change the path
    taken this step. A naive one-shot Euler update (displacement using the START
    speed/heading) makes every primitive land in the same place — the control
    (dtheta, dv) would only bite a step later — which would blind a one-step
    filter. We therefore integrate with `substeps` sub-Euler steps so v and theta
    advance within the interval, closely reproducing the RK45 path.

    Speed is clamped to [v_min, v_max] as the simulator does.
    Returns (new_pos(2,), new_theta, new_speed).
    """
    h = dt / substeps
    p = pos.astype(np.float64).copy()
    th = float(theta)
    v = float(speed)
    for _ in range(substeps):
        p = p + np.array([v * np.cos(th), v * np.sin(th)]) * h
        th = th + dtheta * h
        v = float(np.clip(v + dv * h, v_min, v_max))
    return p.astype(np.float32), th, v


def _rotate(vec2: np.ndarray, heading: float) -> np.ndarray:
    """Reproduce scenario.get_rotated_position_from_relative (world -> ego frame)."""
    c, s = np.cos(heading), np.sin(heading)
    rot = np.array([[c, s], [-s, c]], dtype=np.float32)
    return rot @ vec2.astype(np.float32)


# ---------------------------------------------------------------------------
# The filter
# ---------------------------------------------------------------------------

class CBFSafetyFilter:
    """
    Produces per-agent availability masks over the discrete primitives using the
    frozen barrier network.

    Parameters
    ----------
    barrier : CBFGNN            (frozen, eval mode)
    prim_table : (n_actions,2)  primitive index -> (dtheta, dv)
    sep : float                 separation_distance (km) — for reference/logging
    dt, v_min, v_max : float    air_taxi kinematics constants
    alpha : float               class-K gain in the barrier condition
    eps : float                 slack on the barrier condition
    horizon : int               how many steps ahead to roll each candidate
                                primitive (>=1). At 1 the filter is myopic; a
                                slow-turning vehicle needs a few steps for a
                                primitive's heading/speed change to actually move
                                the aircraft, so 2-4 is recommended. The primitive
                                is HELD constant across the horizon, and every
                                intermediate step must satisfy both the barrier
                                condition and the hard separation floor.
    num_agents : int            number of agent nodes (they lead the entity order)
    neighbour_model : str       how neighbours are propagated over the lookahead:
                                "static" (held at current position — the original
                                decentralised assumption) or "constant_velocity"
                                (advanced along their current heading at their
                                current speed). Constant-velocity is the more
                                realistic model and partially addresses the
                                frozen-neighbour limitation.
    device : torch.device
    """

    def __init__(self, barrier, prim_table, sep, dt, v_min, v_max,
                 alpha=1.0, eps=0.0, horizon=3, fallback="least_unsafe",
                 num_agents=None, neighbour_model="static", device=None,
                 adaptive_alpha=False, alpha_max=2.0, ell_thresh=0.32,
                 ell_hi=0.80, sense_range=None, adaptive_sign=1.0,
                 heading_gate=0.0, adaptive_horizon=False, h_min=1,
                 adaptive_gate="heading", path_gate=0.6,
                 adaptive_stall=False, stall_window=15, stall_thresh=0.15,
                 gap_guard=False):
        self.barrier = barrier
        self.prim_table = prim_table
        self.n_actions = prim_table.shape[0]
        self.sep = sep
        self.dt = dt
        self.v_min = v_min
        self.v_max = v_max
        self.alpha = alpha
        self.eps = eps
        self.horizon = max(1, int(horizon))
        # fallback when no primitive is safe: "least_unsafe" (allow only the
        # safest single move) or "unfiltered" (legacy: allow all).
        self.fallback = fallback
        self.num_agents = num_agents
        assert neighbour_model in ("static", "constant_velocity")
        self.neighbour_model = neighbour_model
        self.device = device if device is not None else torch.device("cpu")

        # --- Rung-1 geometry-adaptive class-K gain (optional, proactive) -------
        # When on, each agent's alpha is ramped UP from `alpha` toward `alpha_max`
        # according to a local conflict-geometry signal (Chandra et al. 2025
        # "liveness" angle between the line-of-approach and the closing velocity).
        # The look-ahead horizon stays fixed at `self.horizon` for Rung 1. The
        # hard separation floor (ok_hard, L~328) is alpha-independent, so a higher
        # alpha only relaxes how aggressively an agent closes WITHIN the safe set;
        # it can never breach `sep`. See Geometry_Adaptive_Filter_Plan.md.
        self.adaptive_alpha = bool(adaptive_alpha)
        self.alpha_max = float(alpha_max)
        self.ell_thresh = float(ell_thresh)   # angle (rad) at/below which ramp is full
        self.ell_hi = float(ell_hi)           # angle (rad) above which ramp is off
        self.sense_range = sense_range        # km; None => every other agent counts
        # +1.0 = corrected sign (converging pair -> SMALL angle); -1.0 flips it.
        self.adaptive_sign = float(adaptive_sign)
        # heading-difference gate (rad): a neighbour whose heading differs from the
        # ego's by LESS than this is treated as in-line following (same stream) and
        # excluded from the ramp, so the corridor segments stay at base alpha and
        # only crossing/opposing geometry (a junction/merge) drives alpha up.
        # 0.0 disables the gate (reproduces the ungated Rung-1 behaviour).
        self.heading_gate = float(heading_gate)
        # in-line / cross-traffic gate selector (which test decides a neighbour is
        # "same-stream following" and so is excluded from driving the ramp):
        #   "heading" (legacy, default): the heading-difference test above. Fails
        #       at a 90 deg bend, where a leader that has already turned has a
        #       heading far from the follower's, so the follower wrongly ramps to
        #       short horizon in the one regime that wants long horizon.
        #   "path": the neighbour's BEARING relative to the ego's own direction of
        #       travel (see _adaptive_alpha). A leader ahead along the (curving)
        #       path stays "ahead" through a bend, so it is correctly excluded,
        #       while cross-traffic at a merge comes in off to the side and still
        #       drives the ramp. path_gate is the half-angle (rad) of the in-line
        #       cone (ahead or behind); reuse the old 0.6 as a sensible default.
        self.adaptive_gate = str(adaptive_gate)
        self.path_gate = float(path_gate)
        # --- junction gap guard (Variant 1) ------------------------------------
        # When on, suppress the adaptive ramp (force severity m=0 -> deployed long
        # horizon, base alpha) for any agent NOT physically inside its corridor
        # tube. The sequential-bend over-ramp lives in the open-space junction
        # between corridors, where a leader that has rounded the 90 deg turn reads
        # as large-heading cross-traffic and wrongly drives the follower to short
        # horizon; the merge convergence happens IN-tube, so the guard leaves it
        # untouched. in_tube is supplied per-agent by the runner from the
        # scenario's own is_in_tube (the SAME test as the in/out-corridor metric).
        self.gap_guard = bool(gap_guard)
        # --- Rung-3 REACTIVE stall trigger (optional) --------------------------
        # When on, the per-agent ramp severity m is NOT taken from the proactive
        # geometry signal at all. Instead an external per-agent PROGRESS-STALL flag
        # (supplied to masks_for_threads as `agent_stalled`, computed from along-
        # corridor progress by the runner) sets m: m=1 for a stalled agent (ramp to
        # the permissive short-horizon / high-alpha regime to break a deadlock),
        # m=0 for a flowing agent (stay at the deployed long horizon / base alpha).
        # This is the discriminator the geometry gates cannot provide: a genuine
        # merge deadlock stalls (-> ramp), a flowing bend does not (-> stay long),
        # so it fixes the sequential-bend regression without the merge collapse the
        # path gate caused. stall_window / stall_thresh are stored here for the
        # runner to read and for the diagnostic print; the stall flag itself is
        # computed runner-side where the corridor frame lives.
        self.adaptive_stall = bool(adaptive_stall)
        self.stall_window = int(stall_window)
        self.stall_thresh = float(stall_thresh)
        # --- Rung-2 geometry-adaptive look-ahead horizon (optional, proactive) --
        # When on, each agent's horizon is ramped DOWN from `self.horizon` toward
        # `h_min` on the SAME conflict-geometry severity m that drives alpha: more
        # convergence -> shorter horizon (less pre-emptive masking -> less freezing
        # at a deadlock-prone junction). Uses the same signal as alpha, so both move
        # toward the permissive (short-h, high-alpha) regime together.
        self.adaptive_horizon = bool(adaptive_horizon)
        self.h_min = max(1, int(h_min))
        # running diagnostics for the mandatory sign-convention / behaviour check
        self.adaptive_stats = dict(calls=0, ramped=0, guard=0, sum_alpha=0.0,
                                   sum_min_ang=0.0, min_min_ang=float("inf"),
                                   max_alpha=float(alpha),
                                   sum_horizon=0.0, min_horizon=int(self.horizon))

    @torch.no_grad()
    def masks_for_threads(self, node_obs, adj, agent_world_states, agent_goals=None,
                          agent_stalled=None, agent_in_tube=None):
        """
        Compute availability masks for every agent in every thread.

        node_obs : (n_threads, num_agents, num_nodes, node_feat)
        adj      : (n_threads, num_agents, num_nodes, num_nodes)  distance matrices
                   (per-agent stack of the shared global distance matrix)
        agent_world_states : (n_threads, num_agents, 4) array of the ego agents'
                   ABSOLUTE world states [x, y, theta, v] this step. This is the
                   ground truth needed to roll a primitive forward; it is read
                   from the env (see the runner hook), NOT reconstructed from the
                   ego-relative node features.

        Returns
        -------
        available : (n_threads, num_agents, n_actions) float32 mask (1 = allowed)
        no_safe   : (n_threads, num_agents) bool — True where NO primitive passed
                    (that agent is left unfiltered; see module docstring)
        """
        n_threads, num_agents = node_obs.shape[0], node_obs.shape[1]
        num_nodes = node_obs.shape[2]
        available = np.ones((n_threads, num_agents, self.n_actions), dtype=np.float32)
        no_safe = np.zeros((n_threads, num_agents), dtype=bool)

        for th in range(n_threads):
            for a in range(num_agents):
                mask, none_safe = self._mask_one_agent(
                    node_obs[th, a], adj[th, a], agent_world_states[th], a, num_nodes,
                    agent_goals_thread=(None if agent_goals is None else agent_goals[th]),
                    agent_stalled_thread=(None if agent_stalled is None else agent_stalled[th]),
                    agent_in_tube_thread=(None if agent_in_tube is None else agent_in_tube[th]))
                available[th, a] = mask
                no_safe[th, a] = none_safe
        return available, no_safe

    def _step_batch(self, pos, theta, speed, dtheta, dv):
        """
        Vectorised one-step air_taxi kinematics for K primitives at once, using
        the same within-step sub-integration as step_air_taxi (so heading/speed
        evolve during the step and bend the path).

        pos:(K,2) theta:(K,) speed:(K,) dtheta:(K,) dv:(K,)  -> new (pos,theta,speed)
        """
        substeps = 20
        h = self.dt / substeps
        p = pos.astype(np.float64).copy()
        th = theta.astype(np.float64).copy()
        v = speed.astype(np.float64).copy()
        for _ in range(substeps):
            p = p + np.stack([v * np.cos(th), v * np.sin(th)], axis=1) * h
            th = th + dtheta * h
            v = np.clip(v + dv * h, self.v_min, self.v_max)
        return p.astype(np.float32), th.astype(np.float32), v.astype(np.float32)

    def _adaptive_alpha(self, ego_pos, ego_vel, ego_theta,
                        nbr_pos, nbr_vel, nbr_theta, ego, ego_goal=None):
        """Rung-1 proactive, conflict-geometry alpha ramp.

        For every other agent j within `sense_range`, measure the angle between
        the line of approach (j relative to the ego) and the closing velocity
        (ego relative to j):

            cos(angle) = <p_j - p_i, v_i - v_j> / (|p_j - p_i| * |v_i - v_j|)

        A pair converging head-on gives angle -> 0 (ego heading straight at j,
        gap shrinking); a diverging / passing pair gives angle -> pi. This is the
        corrected sign of Chandra et al.'s liveness angle: small angle == the
        deadlock-prone, symmetric converging conflict. `adaptive_sign=-1` flips it
        if a future obs convention inverts the geometry (see the sign-check note).

        The most-converging neighbour sets the agent's severity m in [0,1] via a
        linear ramp from `ell_hi` (off) down to `ell_thresh` (full), and alpha is
        ramped continuously alpha -> alpha_max. Continuous = the anti-chatter
        mechanism. Returns (alpha_a, min_angle, m): alpha_a is the ramped gain,
        min_angle is for logging, and m in [0,1] is the raw severity (also used to
        ramp the horizon when adaptive_horizon is on).

        Heading-difference gate (`heading_gate` > 0): a neighbour whose heading is
        within `heading_gate` rad of the ego's is treated as same-stream in-line
        following and excluded, so a trailing aircraft catching up inside a corridor
        does not ramp alpha; only crossing/opposing geometry (a junction/merge)
        does. Headings come from the stored theta (robust when speed ~ 0).
        """
        EPS = 1e-6
        m = 0.0
        min_ang = float("inf")
        na = nbr_pos.shape[0]
        for j in range(na):
            if j == ego:
                continue
            p_rel = nbr_pos[j] - ego_pos          # line of approach (j minus ego)
            d = float(np.linalg.norm(p_rel))
            if self.sense_range is not None and d > self.sense_range:
                continue
            # in-line / cross-traffic gate: exclude same-stream following so that
            # only crossing/opposing geometry (a junction/merge) drives the ramp.
            if self.adaptive_gate == "heading" and self.heading_gate > 0.0:
                # legacy: skip neighbours whose HEADING is close to the ego's.
                dh = float(nbr_theta[j]) - float(ego_theta)
                dh = abs((dh + np.pi) % (2.0 * np.pi) - np.pi)   # wrap to [0, pi]
                if dh < self.heading_gate:
                    continue
            elif self.adaptive_gate == "path" and self.path_gate > 0.0:
                # path-tangent bearing gate: skip neighbours that lie roughly
                # AHEAD or BEHIND along the ego's own direction of travel (in-line
                # flow), keeping only genuine side/cross-traffic. The tangent is
                # the direction to the ego's current waypoint when supplied (it
                # curves with the corridor, so a leader around a bend still reads
                # as "ahead"); otherwise the ego velocity (fine on straights,
                # weaker only in the ~1-2 steps right at a corner apex).
                tan = None
                if ego_goal is not None:
                    tan = np.asarray(ego_goal, dtype=np.float32) - ego_pos
                    if float(np.linalg.norm(tan)) < EPS:
                        tan = None
                if tan is None:
                    tan = ego_vel
                tn = float(np.linalg.norm(tan))
                if tn >= EPS:                       # no heading at all => don't gate
                    tan = tan / tn
                    bn = p_rel / (d + EPS)          # unit bearing to neighbour j
                    cosb = max(-1.0, min(1.0, float(np.dot(tan, bn))))
                    rel_bearing = float(np.arccos(cosb))   # 0 = ahead, pi = behind
                    if (rel_bearing < self.path_gate
                            or rel_bearing > (np.pi - self.path_gate)):
                        continue                    # in-line flow -> keep long horizon
            v_rel = ego_vel - nbr_vel[j]          # closing velocity (ego minus j)
            den = d * float(np.linalg.norm(v_rel)) + EPS
            c = self.adaptive_sign * float(np.dot(p_rel, v_rel)) / den
            c = max(-1.0, min(1.0, c))
            ang = float(np.arccos(c))             # small => converging / deadlock-prone
            if ang < min_ang:
                min_ang = ang
            sev = (self.ell_hi - ang) / (self.ell_hi - self.ell_thresh + EPS)
            sev = max(0.0, min(1.0, sev))
            if sev > m:
                m = sev
        alpha_a = self.alpha + (self.alpha_max - self.alpha) * m
        return alpha_a, min_ang, m

    def _mask_one_agent(self, node_obs_a, adj_a, world_states_thread, ego, num_nodes,
                        agent_goals_thread=None, agent_stalled_thread=None,
                        agent_in_tube_thread=None):
        """
        node_obs_a : (num_nodes, node_feat)  ego agent `ego`'s node features
        adj_a      : (num_nodes, num_nodes)  the (shared) distance matrix
        world_states_thread : (num_agents, 4) absolute [x,y,theta,v] for all agents
        ego        : int, this agent's node index
        """
        ego_state = world_states_thread[ego]
        other_agent_pos = world_states_thread[:, :2].astype(np.float32)  # (na, 2)
        na = other_agent_pos.shape[0]
        K = self.n_actions

        # neighbour velocity vectors (only used by the constant-velocity model):
        # v * [cos(theta), sin(theta)] per agent, from the absolute world state.
        nbr_theta = world_states_thread[:, 2].astype(np.float32)
        nbr_speed = world_states_thread[:, 3].astype(np.float32)
        nbr_vel = np.stack([nbr_speed * np.cos(nbr_theta),
                            nbr_speed * np.sin(nbr_theta)], axis=1)        # (na, 2)

        # per-agent class-K gain: constant (= self.alpha) unless the geometry-
        # adaptive ramp is enabled, in which case it is raised from self.alpha
        # toward self.alpha_max by this agent's local conflict-geometry signal.
        if self.adaptive_alpha or self.adaptive_horizon:
            if self.adaptive_stall:
                # Rung 3: severity is driven by the external progress-stall flag,
                # NOT by geometry. 1 (stalled) -> ramp to short-h/high-alpha to
                # break the deadlock; 0 (flowing) -> stay at the deployed long
                # horizon / base alpha. A flowing bend never stalls, so it is not
                # ramped (fixes the sequential regression); a merge deadlock does.
                _stalled = (agent_stalled_thread is not None
                            and bool(agent_stalled_thread[ego]))
                _m = 1.0 if _stalled else 0.0
                _min_ang = float("nan")
                alpha_a = self.alpha + (self.alpha_max - self.alpha) * _m
            else:
                ego_vel = np.array([ego_state[3] * np.cos(ego_state[2]),
                                    ego_state[3] * np.sin(ego_state[2])],
                                   dtype=np.float32)                        # (2,)
                alpha_a, _min_ang, _m = self._adaptive_alpha(
                    ego_state[:2].astype(np.float32), ego_vel, float(ego_state[2]),
                    other_agent_pos, nbr_vel, nbr_theta, ego,
                    ego_goal=(None if agent_goals_thread is None
                              else agent_goals_thread[ego]))
            # junction gap guard: outside the corridor, cancel the ramp (force
            # m=0 -> deployed long horizon + base alpha). See __init__ gap_guard.
            if (self.gap_guard and agent_in_tube_thread is not None
                    and not bool(agent_in_tube_thread[ego])):
                _m = 0.0
                alpha_a = self.alpha
                self.adaptive_stats["guard"] += 1
            alpha_use = alpha_a if self.adaptive_alpha else self.alpha
            # Rung-2: ramp the per-agent horizon DOWN with the same severity m
            # (m=0 -> deployed horizon; m=1 -> h_min). Rounded to an integer.
            if self.adaptive_horizon:
                hz_use = int(max(self.h_min,
                                 round(self.horizon - (self.horizon - self.h_min) * _m)))
            else:
                hz_use = self.horizon
            st = self.adaptive_stats
            st["calls"] += 1
            st["sum_alpha"] += alpha_use
            if alpha_use > self.alpha + 1e-3:
                st["ramped"] += 1
            if np.isfinite(_min_ang):
                st["sum_min_ang"] += _min_ang
                st["min_min_ang"] = min(st["min_min_ang"], _min_ang)
            st["max_alpha"] = max(st["max_alpha"], alpha_use)
            st["sum_horizon"] += hz_use
            st["min_horizon"] = min(st["min_horizon"], hz_use)
        else:
            alpha_use = self.alpha
            hz_use = self.horizon

        # current barrier value for the ego (baseline h at t=0)
        no_t = torch.as_tensor(node_obs_a[None], dtype=torch.float32, device=self.device)
        adj_t = torch.as_tensor(adj_a[None], dtype=torch.float32, device=self.device)
        h_prev = np.full(K, float(self.barrier(no_t, adj_t)[0, ego]), dtype=np.float32)

        # Per-primitive rolled ego state, held primitive constant across horizon.
        pos = np.repeat(ego_state[None, :2].astype(np.float32), K, axis=0)  # (K,2)
        theta = np.repeat(np.float32(ego_state[2]), K)                      # (K,)
        speed = np.repeat(np.float32(ego_state[3]), K)                      # (K,)

        # candidate graphs share the base node_obs / adj; only the ego row of adj
        # changes as the ego (and, under constant_velocity, the neighbours) roll
        # forward.
        cand_node_obs = np.repeat(node_obs_a[None], K, axis=0)              # (K,N,F)
        base_adj = np.repeat(adj_a[None], K, axis=0)                        # (K,N,N)

        keep = np.ones(K, dtype=bool)          # passes every step so far
        min_dist_overall = np.full(K, np.inf, dtype=np.float32)

        dth = self.prim_table[:, 0].astype(np.float32)   # (K,)
        dv = self.prim_table[:, 1].astype(np.float32)    # (K,)

        for t in range(hz_use):
            # advance every primitive's ego one step (vectorised kinematics)
            pos, theta, speed = self._step_batch(pos, theta, speed, dth, dv)

            # neighbour positions at this step: static holds them; constant_velocity
            # advances each neighbour by its velocity over the elapsed time (t+1)*dt
            # (straight-line, no turning — the standard constant-velocity model).
            if self.neighbour_model == "constant_velocity":
                nbr_pos_t = other_agent_pos + nbr_vel * (self.dt * (t + 1))  # (na,2)
            else:
                nbr_pos_t = other_agent_pos

            # ego-to-other-agent distances after this step, per primitive
            # pos: (K,2), nbr_pos_t: (na,2) -> (K, na)
            diff = pos[:, None, :] - nbr_pos_t[None, :, :]
            dists = np.linalg.norm(diff, axis=2).astype(np.float32)         # (K, na)

            # write ego row/col of adj for the barrier eval
            cand_adj = base_adj.copy()
            cand_adj[:, ego, :na] = dists
            cand_adj[:, :na, ego] = dists
            cand_adj[np.arange(K), ego, ego] = 0.0

            # nearest OTHER agent this step (ignore self column)
            d_others = dists.copy()
            d_others[:, ego] = np.inf
            step_min = d_others.min(axis=1)                                # (K,)
            min_dist_overall = np.minimum(min_dist_overall, step_min)

            # barrier at this step
            cand_no = torch.as_tensor(cand_node_obs, dtype=torch.float32, device=self.device)
            cand_ad = torch.as_tensor(cand_adj, dtype=torch.float32, device=self.device)
            h_t = self.barrier(cand_no, cand_ad)[:, ego].cpu().numpy()     # (K,)

            # step-to-step barrier condition: h_t - h_prev + alpha*h_prev >= -eps
            # (alpha_use == self.alpha unless the geometry-adaptive ramp is on)
            ok_barrier = (h_t - h_prev + alpha_use * h_prev) >= -self.eps
            ok_hard = step_min >= self.sep
            keep &= ok_barrier & ok_hard

            h_prev = h_t
            # (once a primitive fails, it stays failed; we keep rolling the rest
            #  together for vectorisation simplicity — cheap at K=25.)

        keep_barrier = keep                    # already folds in the barrier test
        min_dist_after = min_dist_overall
        # (2) hard separation floor across the whole horizon (already applied via
        #     ok_hard each step; kept here as an explicit final guard).
        keep_hard = min_dist_after >= self.sep
        keep = keep_barrier & keep_hard  # (K,) bool

        if keep.any():
            return keep.astype(np.float32), False

        # --- No primitive passes both tests: LEAST-UNSAFE fallback ---
        # Rather than surrendering (all primitives available, which lets the
        # policy do anything and was found to make dense conflicts worse), allow
        # only the single primitive that keeps the ego FURTHEST from any other
        # aircraft over the horizon (largest min-distance). This is the safest
        # available move and keeps the filter in control even when no move is
        # strictly "safe". Still flagged as a no-safe-action step for logging.
        if self.fallback == "least_unsafe":
            best = int(np.argmax(min_dist_after))
            mask = np.zeros(self.n_actions, dtype=np.float32)
            mask[best] = 1.0
            return mask, True
        # legacy behaviour: leave unfiltered
        return np.ones(self.n_actions, dtype=np.float32), True


# ============================= ego-frame (training) ===========================
#
# Ego-frame Control-Barrier-Function (CBF) safety filter.
#
# Runs entirely on the in-process graph data the policy already receives
# (node_obs / adj / obs / agent_id), so it needs no access to world.agents and
# works unchanged under subprocess-vectorised training. Validated to reproduce
# the world-frame geometry exactly (see the validation note in the module
# docstring). One-step lookahead (horizon = 1): for every motion primitive it
# rolls the ego one step, rebuilds the ego's candidate graph, and keeps the
# primitive iff the learned barrier still certifies the next state
# (h_next - h_now + alpha*h_now >= -eps) AND the predicted nearest-neighbour
# distance stays >= the separation minimum. If no primitive passes, the
# least-unsafe fallback keeps the single primitive that maximises the predicted
# nearest-neighbour distance.
#
# Node-feature layout (per ego graph): node_obs[:, :, 0:2] = relative velocity,
# node_obs[:, :, 2:4] = relative position; obs[:, 2] = ego speed; agent_id = the
# ego's own node index (the identifier the policy GNN itself uses).


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
