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

  * EgoCBFFilter     (ego frame)   — used during TRAINING (Stages 3A/3B). Reads
    only the in-process graph tensors the policy already receives (node_obs /
    adj / obs / agent_id), so it needs no access to world.agents and works under
    subprocess-vectorised training. One-step lookahead (horizon = 1).

Both apply the same safety rule (barrier condition AND hard separation floor,
least-unsafe fallback). They differ in the kinematic prediction: the ego-frame
filter takes ONE step using the end-of-step speed/heading, while the world-frame
filter sub-integrates each step (20 sub-steps) over a multi-step horizon. The
predicted positions therefore differ slightly (up to ~3.7 m per step, ~2.4% of
the 152 m separation, over all 25 primitives). They were kept in separate files
during development and are consolidated here; the shared primitive table
(build_primitive_table) is defined once, below, and used by both.

================================ world-frame (eval) ===========================

CBF safety filter (Stage 2 of Option B, extended with the adaptive ramp).

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
  2. Roll the ego aircraft forward with the air_taxi kinematics
     (multiagent/core.AirTaxiXYState.dstate), holding the primitive constant for
     `horizon` steps. Neighbours are propagated according to `neighbour_model`:
     "static" holds them at their current position; "constant_velocity" advances
     them along their current heading at their current speed.
  3. At every step, rebuild the ego's row/column of the distance matrix `adj`
     from the predicted positions (the separation signal the barrier keys on).
     Node features are NOT re-predicted: the barrier sees the predicted
     distances with the current relative velocities.
  4. Evaluate the barrier at the predicted graph: h_t = barrier(...)[ego].
  5. Keep the primitive iff, at EVERY step of the horizon, the discrete barrier
     condition holds
         h_t - h_{t-1} + alpha * h_{t-1} >= -eps
     (the one-step form of h_dot + alpha*h >= 0) AND the predicted nearest-
     neighbour distance stays >= the separation minimum (hard floor).

If NO primitive passes for an agent, the mask would be all-zero, which the
Categorical cannot handle. The default `least_unsafe` fallback then allows only
the single primitive that keeps the ego furthest from any other aircraft over
the horizon; the legacy `unfiltered` fallback (kept for the fallback ablation)
allows every primitive. Either way the step is flagged, so the "no safe
primitive available" rate can be logged.

Optional adaptive layer (eval only): a per-agent severity m in [0, 1], computed
from the conflict geometry (liveness angle, see _adaptive_alpha), ramps alpha up
(alpha -> alpha_max) and the horizon down (horizon -> h_min). The junction gap
guard forces m = 0 for any agent outside its corridor tube.

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
                 ell_hi=0.80, sense_range=None,
                 heading_gate=0.0, adaptive_horizon=False, h_min=1,
                 gap_guard=False, ignore_parked=False, horizon_out=0,
                 route_turn_limit=0.0, route_lookahead=0.2, route_variant="orig"):
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
        # [ES] added: "route" = predict each neighbour along its own corridor route
        # (see route_paths / _route_predict below); falls back to constant velocity
        # for any neighbour without a usable route. Default model is unchanged.
        assert neighbour_model in ("static", "constant_velocity", "route", "route_map")
        self.neighbour_model = neighbour_model
        self.route_stats = {"route": 0, "cv_fallback": 0}
        # [ES] added: turn-rate-limited route following (rad/s; 0 = off). Off: the
        # predicted neighbour moves along its route polyline, so it takes the
        # route's direction at once (at the start and at every corner). On: it
        # steers towards the point route_lookahead (km) ahead of it on the route
        # (pure pursuit) and can turn no faster than route_turn_limit, so the
        # predicted path arcs through corners as a real aircraft must.
        self.route_turn_limit = float(route_turn_limit or 0.0)
        self.route_lookahead = float(route_lookahead)
        # [ES] added: how the route / route_map models predict (default "orig" = as
        # first built). "fix": also follow the route in the aircraft's final corridor
        # ("orig" needs >= 2 route waypoints, so a neighbour in its final corridor,
        # whose route is just its exit, was predicted by constant velocity).
        # "hybrid": "fix" inside the neighbour's current corridor; outside it (queue,
        # junction gap, past the final exit) steer towards the neighbour's current
        # goal, turn rate capped at the air-taxi limit. "goal": always steer towards
        # the goal. Goals: the simulator's goal ("route") or the published goal point
        # of the inferred corridor ("route_map").
        assert route_variant in ("orig", "fix", "hybrid", "goal")
        self.route_variant = route_variant
        self.pred_turn_max = float(np.max(np.abs(prim_table[:, 0])))
        self.device = device if device is not None else torch.device("cpu")

        # --- Rung-1 geometry-adaptive class-K gain (optional, proactive) -------
        # When on, each agent's alpha is ramped UP from `alpha` toward `alpha_max`
        # according to a local conflict-geometry signal (Chandra et al. 2025
        # "liveness" angle between the line-of-approach and the closing velocity).
        # The look-ahead horizon stays fixed at `self.horizon` for Rung 1. The
        # hard separation floor (ok_hard in _mask_one_agent) is alpha-independent,
        # so a higher alpha only relaxes how aggressively an agent closes WITHIN
        # the safe set; it can never breach `sep`. See Geometry_Adaptive_Filter_Plan.md.
        self.adaptive_alpha = bool(adaptive_alpha)
        self.alpha_max = float(alpha_max)
        self.ell_thresh = float(ell_thresh)   # angle (rad) at/below which ramp is full
        self.ell_hi = float(ell_hi)           # angle (rad) above which ramp is off
        self.sense_range = sense_range        # km; None => every other agent counts
        # heading-difference gate (rad): a neighbour whose heading differs from the
        # ego's by LESS than this is treated as in-line following (same stream) and
        # excluded from the ramp, so the corridor segments stay at base alpha and
        # only crossing/opposing geometry (a junction/merge) drives alpha up.
        # 0.0 disables the gate (reproduces the ungated Rung-1 behaviour).
        # Known weakness: at a 90 deg bend a leader that has already turned has a
        # heading far from the follower's, so the gate lets it through and the
        # follower wrongly ramps to short horizon; the junction gap guard below
        # removes that case.
        self.heading_gate = float(heading_gate)
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
        # --- parked-aircraft exclusion (optional) -------------------------------
        # When on (and the runner supplies agent_active), aircraft that have
        # reached their goal (agent.status True) are dropped from the ego's
        # neighbour set: the separation test, the barrier's candidate graph and
        # the adaptive ramp. This matches the environment, which already treats
        # them as gone: it skips collisions involving them, disconnects them from
        # the policy's graph (adj rows/cols zeroed) and drops them from the
        # nearest-neighbour features. Without it, every later arrival must keep
        # 152 m from the aircraft parked around the SHARED goal point, and the
        # parked aircraft carry a phantom velocity (reset_velocity() at the goal
        # sets speed = v_min and a random heading, which the constant-velocity
        # model extrapolates). A parked EGO gets an all-allowed mask (the env
        # freezes it) and is left out of every diagnostic. Off by default so
        # earlier results reproduce exactly.
        self.ignore_parked = bool(ignore_parked)
        # --- longer look-ahead outside the corridor (optional) -------------------
        # When > 0 (and the runner supplies agent_in_tube), an ego that is NOT
        # physically inside its corridor tube uses this horizon instead of
        # `horizon`. Motivation: on the sequential bend most residual losses of
        # separation are in the open junction gap, where aircraft turn 90 deg and
        # the filter usually had no safe move left on the step of the loss, while
        # a longer look-ahead everywhere costs success at the merges (in-tube).
        # 0 = off (default), so earlier results reproduce exactly.
        self.horizon_out = max(0, int(horizon_out))
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
        # test-binding diagnostic (read-only: counts which test blocks each move;
        # never changes a mask). Split by regime: "calm" = base alpha and full
        # horizon, "ramped" = the adaptive layer relaxed alpha and/or the horizon.
        # Only ACTIVE aircraft are counted when the runner supplies agent_active:
        # aircraft parked at their goal share one goal point per exit corridor, so
        # including them would inflate the blocked / no-safe counts.
        self.bind_stats = {r: dict(steps=0, checks=0, passed=0, barrier_only=0,
                                   separation_only=0, both=0, steps_barrier_extra=0,
                                   steps_barrier_causes_none=0, steps_none=0,
                                   steps_outside_tube=0)
                           for r in ("calm", "ramped")}
        self.bind_skipped_inactive = 0

    @torch.no_grad()
    def masks_for_threads(self, node_obs, adj, agent_world_states, agent_in_tube=None,
                          agent_active=None, agent_routes=None):
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
        agent_in_tube : (n_threads, num_agents) bool or None — per-agent "physically
                   inside its corridor tube" flag for the junction gap guard
                   (ignored unless gap_guard is on).
        agent_routes : list (per thread) of per-agent route polylines (arrays of
                   waypoints, km) or None — used only by neighbour_model "route"
                   (see route_paths); None means constant velocity for everyone.
        agent_active : (n_threads, num_agents) bool or None — False for aircraft
                   parked at their goal. By default used ONLY to keep them out of
                   the test-binding diagnostic (never changes a mask); with
                   ignore_parked on it also drops them from every ego's
                   neighbour set (see __init__).

        Returns
        -------
        available : (n_threads, num_agents, n_actions) float32 mask (1 = allowed)
        no_safe   : (n_threads, num_agents) bool — True where NO primitive passed
                    (the fallback was applied; see module docstring)
        """
        n_threads, num_agents = node_obs.shape[0], node_obs.shape[1]
        num_nodes = node_obs.shape[2]
        available = np.ones((n_threads, num_agents, self.n_actions), dtype=np.float32)
        no_safe = np.zeros((n_threads, num_agents), dtype=bool)

        for th in range(n_threads):
            # [ES] added: route model -- predict every neighbour once per thread and step
            # (shared by all egos), for the longest look-ahead any ego can use.
            self._route_pred_cache = None
            if self.neighbour_model in ("route", "route_map"):
                hz_max = max(int(self.horizon), int(getattr(self, "horizon_out", 0) or 0))
                self._route_pred_cache = self._route_predict(
                    agent_world_states[th], None if agent_routes is None else agent_routes[th], hz_max)
            for a in range(num_agents):
                mask, none_safe = self._mask_one_agent(
                    node_obs[th, a], adj[th, a], agent_world_states[th], a, num_nodes,
                    agent_in_tube_thread=(None if agent_in_tube is None else agent_in_tube[th]),
                    agent_active_thread=(None if agent_active is None else agent_active[th]),
                    agent_routes_thread=(None if agent_routes is None else agent_routes[th]))
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

    def _record_binding(self, bar_ok, hard_ok, regime, outside_tube=False):
        """Read-only bookkeeping for the test-binding diagnostic. For each of the
        K moves: passed both tests, blocked by the barrier test only, by the
        separation test only, or by both. Also counts the agent-steps where the
        barrier removed at least one move the separation test allowed, and where
        the barrier alone left no move (separation test alone would have)."""
        st = self.bind_stats[regime]
        st["steps"] += 1
        st["checks"] += int(bar_ok.size)
        st["passed"] += int((bar_ok & hard_ok).sum())
        st["barrier_only"] += int((~bar_ok & hard_ok).sum())
        st["separation_only"] += int((bar_ok & ~hard_ok).sum())
        st["both"] += int((~bar_ok & ~hard_ok).sum())
        if (~bar_ok & hard_ok).any():
            st["steps_barrier_extra"] += 1
        if hard_ok.any() and not (bar_ok & hard_ok).any():
            st["steps_barrier_causes_none"] += 1
        if not (bar_ok & hard_ok).any():
            st["steps_none"] += 1
        if outside_tube:
            st["steps_outside_tube"] += 1

    def _adaptive_alpha(self, ego_pos, ego_vel, ego_theta,
                        nbr_pos, nbr_vel, nbr_theta, ego, skip=None):
        """Rung-1 proactive, conflict-geometry alpha ramp.

        For every other agent j within `sense_range`, measure the angle between
        the line of approach (j relative to the ego) and the closing velocity
        (ego relative to j):

            cos(angle) = <p_j - p_i, v_i - v_j> / (|p_j - p_i| * |v_i - v_j|)

        A pair converging head-on gives angle -> 0 (ego heading straight at j,
        gap shrinking); a diverging / passing pair gives angle -> pi. This is the
        corrected sign of Chandra et al.'s liveness angle: small angle == the
        deadlock-prone, symmetric converging conflict (sign convention verified
        in-env: merge ramps far more than free flow).

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
            if skip is not None and skip[j]:      # parked aircraft (ignore_parked)
                continue
            p_rel = nbr_pos[j] - ego_pos          # line of approach (j minus ego)
            d = float(np.linalg.norm(p_rel))
            if self.sense_range is not None and d > self.sense_range:
                continue
            # in-line / cross-traffic gate: exclude same-stream following so that
            # only crossing/opposing geometry (a junction/merge) drives the ramp.
            if self.heading_gate > 0.0:
                # skip neighbours whose HEADING is close to the ego's.
                dh = float(nbr_theta[j]) - float(ego_theta)
                dh = abs((dh + np.pi) % (2.0 * np.pi) - np.pi)   # wrap to [0, pi]
                if dh < self.heading_gate:
                    continue
            v_rel = ego_vel - nbr_vel[j]          # closing velocity (ego minus j)
            den = d * float(np.linalg.norm(v_rel)) + EPS
            c = float(np.dot(p_rel, v_rel)) / den
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

    # [ES] added: route-aware neighbour prediction ------------------------------
    ROUTE_MAX_HEADING_DEG = 60.0   # use the route only if the neighbour flies roughly along it

    def _route_predict(self, world_states_thread, routes, hz):
        """Predicted neighbour positions (hz, na, 2): each neighbour advances at its
        current speed along its route polyline (current corridor, then the next
        ones); constant velocity if it has no route or is not flying along it
        (heading more than ROUTE_MAX_HEADING_DEG off the route direction)."""
        ws = np.asarray(world_states_thread, dtype=np.float64)
        na = ws.shape[0]
        out = np.empty((hz, na, 2), dtype=np.float32)
        steps = self.dt * np.arange(1, hz + 1)
        variant = getattr(self, "route_variant", "orig")
        goals = getattr(routes, "goals", None)
        inside = getattr(routes, "inside", None)
        min_wp = 2 if variant == "orig" else 1
        for j in range(na):
            p, th, v = ws[j, :2], ws[j, 2], ws[j, 3]
            vel = v * np.array([np.cos(th), np.sin(th)])
            path = None if routes is None else routes[j]
            used = False
            if (variant in ("hybrid", "goal") and goals is not None and goals[j] is not None
                    and (variant == "goal" or not inside[j])):
                out[:, j] = _pursue_point_turn_limited(p, th, v, goals[j], self.dt, hz, self.pred_turn_max)
                self.route_stats["route"] += 1
                continue
            if path is not None and len(path) >= min_wp:
                pts = np.vstack([p[None], np.asarray(path, dtype=np.float64)])
                seg = np.diff(pts, axis=0)
                L = np.linalg.norm(seg, axis=1)
                keep = L > 1e-6
                pts = np.vstack([pts[:1], pts[1:][keep]]); seg = seg[keep]; L = L[keep]
                if len(L):
                    d0 = seg[0] / L[0]
                    hd = np.array([np.cos(th), np.sin(th)])
                    if float(np.dot(d0, hd)) >= np.cos(np.radians(self.ROUTE_MAX_HEADING_DEG)):
                        cum = np.concatenate([[0.0], np.cumsum(L)])
                        if self.route_turn_limit > 0:
                            out[:, j] = _follow_turn_limited(p, th, v, pts, seg, L, cum, self.dt, hz,
                                                             self.route_turn_limit, self.route_lookahead)
                            used = True
                            self.route_stats["route"] += 1
                            continue
                        for k, dist in enumerate(v * steps):
                            if dist >= cum[-1]:          # beyond the last waypoint: carry on straight
                                out[k, j] = pts[-1] + (dist - cum[-1]) * seg[-1] / L[-1]
                            else:
                                q = int(np.searchsorted(cum, dist, side="right") - 1)
                                out[k, j] = pts[q] + (dist - cum[q]) * seg[q] / L[q]
                        used = True
            if not used:
                out[:, j] = p[None] + vel[None] * steps[:, None]
            self.route_stats["route" if used else "cv_fallback"] += 1
        return out

    def _mask_one_agent(self, node_obs_a, adj_a, world_states_thread, ego, num_nodes,
                        agent_in_tube_thread=None, agent_active_thread=None,
                        agent_routes_thread=None):
        """
        node_obs_a : (num_nodes, node_feat)  ego agent `ego`'s node features
        adj_a      : (num_nodes, num_nodes)  the (shared) distance matrix
        world_states_thread : (num_agents, 4) absolute [x,y,theta,v] for all agents
        ego        : int, this agent's node index
        agent_in_tube_thread : (num_agents,) bool or None — gap-guard input
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

        # parked-aircraft exclusion (see __init__ ignore_parked). `parked` is a
        # (na,) bool mask of aircraft to drop from the neighbour set, or None.
        parked = None
        if self.ignore_parked and agent_active_thread is not None:
            _act = np.asarray(agent_active_thread, dtype=bool)
            if not _act[ego]:
                # the ego itself has parked: the env freezes it, so its mask has
                # no effect -- allow everything and keep it out of the stats.
                self.bind_skipped_inactive += 1
                return np.ones(self.n_actions, dtype=np.float32), False
            parked = ~_act
            parked_idx = np.flatnonzero(parked)

        # per-agent class-K gain: constant (= self.alpha) unless the geometry-
        # adaptive ramp is enabled, in which case it is raised from self.alpha
        # toward self.alpha_max by this agent's local conflict-geometry signal.
        if self.adaptive_alpha or self.adaptive_horizon:
            ego_vel = np.array([ego_state[3] * np.cos(ego_state[2]),
                                ego_state[3] * np.sin(ego_state[2])],
                               dtype=np.float32)                            # (2,)
            alpha_a, _min_ang, _m = self._adaptive_alpha(
                ego_state[:2].astype(np.float32), ego_vel, float(ego_state[2]),
                other_agent_pos, nbr_vel, nbr_theta, ego, skip=parked)
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
            if (self.horizon_out > 0 and agent_in_tube_thread is not None
                    and not bool(agent_in_tube_thread[ego])):
                hz_use = self.horizon_out

        # [ES] added: barrier=None -> no learned barrier at all (hard separation
        # test only); the barrier network is never evaluated.
        use_barrier = self.barrier is not None
        # current barrier value for the ego (baseline h at t=0)
        if use_barrier:
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
        bar_ok = np.ones(K, dtype=bool)        # diagnostic: barrier test alone
        hard_ok = np.ones(K, dtype=bool)       # diagnostic: separation test alone
        min_dist_overall = np.full(K, np.inf, dtype=np.float32)

        dth = self.prim_table[:, 0].astype(np.float32)   # (K,)
        dv = self.prim_table[:, 1].astype(np.float32)    # (K,)

        # [ES] added: route model -- every neighbour's predicted position at each
        # look-ahead step along its own corridor route (constant velocity where no
        # usable route). Not used by the other models.
        nbr_route_pred = None
        if self.neighbour_model in ("route", "route_map"):
            cache = getattr(self, "_route_pred_cache", None)
            if cache is not None and cache.shape[0] >= hz_use:
                nbr_route_pred = cache[:hz_use]                               # (hz, na, 2)
            else:
                nbr_route_pred = self._route_predict(world_states_thread, agent_routes_thread,
                                                     hz_use)


        for t in range(hz_use):
            # advance every primitive's ego one step (vectorised kinematics)
            pos, theta, speed = self._step_batch(pos, theta, speed, dth, dv)

            # neighbour positions at this step: static holds them; constant_velocity
            # advances each neighbour by its velocity over the elapsed time (t+1)*dt
            # (straight-line, no turning — the standard constant-velocity model).
            if self.neighbour_model == "constant_velocity":
                nbr_pos_t = other_agent_pos + nbr_vel * (self.dt * (t + 1))  # (na,2)
            elif nbr_route_pred is not None:
                nbr_pos_t = nbr_route_pred[t]                                 # (na,2)
            else:
                nbr_pos_t = other_agent_pos

            # ego-to-other-agent distances after this step, per primitive
            # pos: (K,2), nbr_pos_t: (na,2) -> (K, na)
            diff = pos[:, None, :] - nbr_pos_t[None, :, :]
            dists = np.linalg.norm(diff, axis=2).astype(np.float32)         # (K, na)

            # write ego row/col of adj for the barrier eval
            if use_barrier:
                cand_adj = base_adj.copy()
                cand_adj[:, ego, :na] = dists
                cand_adj[:, :na, ego] = dists
                cand_adj[np.arange(K), ego, ego] = 0.0
                if parked is not None and parked_idx.size:
                    # keep parked aircraft disconnected (adj 0 = no edge), exactly as
                    # in the env's own graph used for h at t=0.
                    cand_adj[:, ego, parked_idx] = 0.0
                    cand_adj[:, parked_idx, ego] = 0.0

            # nearest OTHER agent this step (ignore self column)
            d_others = dists.copy()
            d_others[:, ego] = np.inf
            if parked is not None:
                d_others[:, parked] = np.inf          # parked aircraft are not traffic
            step_min = d_others.min(axis=1)                                # (K,)
            min_dist_overall = np.minimum(min_dist_overall, step_min)

            # barrier at this step
            if use_barrier:
                cand_no = torch.as_tensor(cand_node_obs, dtype=torch.float32, device=self.device)
                cand_ad = torch.as_tensor(cand_adj, dtype=torch.float32, device=self.device)
                h_t = self.barrier(cand_no, cand_ad)[:, ego].cpu().numpy()     # (K,)

                # step-to-step barrier condition: h_t - h_prev + alpha*h_prev >= -eps
                # (alpha_use == self.alpha unless the geometry-adaptive ramp is on)
                ok_barrier = (h_t - h_prev + alpha_use * h_prev) >= -self.eps
            else:
                ok_barrier = np.ones(K, dtype=bool)
            ok_hard = step_min >= self.sep
            keep &= ok_barrier & ok_hard
            bar_ok &= ok_barrier
            hard_ok &= ok_hard

            if use_barrier:
                h_prev = h_t
            # (once a primitive fails, it stays failed; we keep rolling the rest
            #  together for vectorisation simplicity — cheap at K=25.)

        keep_barrier = keep                    # already folds in the barrier test
        min_dist_after = min_dist_overall
        # (2) hard separation floor across the whole horizon (already applied via
        #     ok_hard each step; kept here as an explicit final guard).
        keep_hard = min_dist_after >= self.sep
        keep = keep_barrier & keep_hard  # (K,) bool
        if agent_active_thread is None or bool(agent_active_thread[ego]):
            self._record_binding(
                bar_ok, hard_ok & keep_hard,
                "ramped" if (alpha_use > self.alpha + 1e-3 or hz_use != self.horizon) else "calm",
                outside_tube=(agent_in_tube_thread is not None
                              and not bool(agent_in_tube_thread[ego])))
        else:
            self.bind_skipped_inactive += 1

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


# [ES] added: route polylines for the "route" neighbour model ----------------------
def _follow_turn_limited(p, th, v, pts, seg, L, cum, dt, hz, wmax, lookahead, substeps=10):
    """[ES] added: predicted positions (hz, 2) of an aircraft at p (heading th, speed v)
    that follows the route polyline pts by pure pursuit: each substep it steers towards
    the route point `lookahead` km beyond its own progress along the route, with its
    turn rate capped at wmax (rad/s); speed is held. Beyond the last waypoint the
    target carries on along the last segment."""
    def at(s):
        if s >= cum[-1]:
            return pts[-1] + (s - cum[-1]) * seg[-1] / L[-1]
        q = int(np.searchsorted(cum, s, side="right") - 1)
        return pts[q] + (s - cum[q]) * seg[q] / L[q]
    out = np.empty((hz, 2), dtype=np.float64)
    q = np.array(p, dtype=np.float64)
    th = float(th)
    h = dt / substeps
    s = 0.0
    for k in range(hz):
        for _ in range(substeps):
            s += v * h
            tgt = at(s + lookahead)
            err = (np.arctan2(tgt[1] - q[1], tgt[0] - q[0]) - th + np.pi) % (2 * np.pi) - np.pi
            th += float(np.clip(err, -wmax * h, wmax * h))
            q = q + v * h * np.array([np.cos(th), np.sin(th)])
        out[k] = q
    return out


def _pursue_point_turn_limited(p, th, v, goal, dt, hz, wmax, substeps=10):
    """[ES] added: predicted positions (hz, 2) of an aircraft at p (heading th, speed v)
    that steers straight at `goal` with its turn rate capped at wmax (rad/s); speed held."""
    out = np.empty((hz, 2), dtype=np.float64)
    q = np.array(p, dtype=np.float64)
    g = np.asarray(goal, dtype=np.float64)
    th = float(th)
    h = dt / substeps
    for k in range(hz):
        for _ in range(substeps):
            err = (np.arctan2(g[1] - q[1], g[0] - q[0]) - th + np.pi) % (2 * np.pi) - np.pi
            th += float(np.clip(err, -wmax * h, wmax * h))
            q = q + v * h * np.array([np.cos(th), np.sin(th)])
        out[k] = q
    return out


class RouteSet(list):
    """[ES] added: per-aircraft route polylines (the list itself, unchanged), plus
    .goals (per aircraft goal point or None) and .inside (inside its current corridor),
    used only by the "hybrid" / "goal" route variants."""
    goals = None
    inside = None


def _published_goal(scen, world, k):
    """Goal point of corridor k from the published layout (where the scenario puts the
    goal for aircraft in that corridor); None if the scenario has no such rule."""
    tp = world.tube_params
    tubes = [tp] if isinstance(tp, dict) else list(tp)
    if hasattr(scen, "_goal_pos_for_tube"):
        return np.asarray(scen._goal_pos_for_tube(tubes[k]), dtype=float)
    lp = np.asarray(getattr(scen, "landmark_poses", []), dtype=float)
    if len(lp) and np.allclose(lp, lp[0]):          # single corridor: one shared goal point
        return lp[0].copy()
    return None


def _map_goal(scen, world, i, rem, gap_reach=1.2):
    """Goal of aircraft i from sensing + the published map: the goal of its inferred
    current corridor; if it is just past the exit of a final corridor, that corridor's
    goal (the destination); else None."""
    if rem:
        return _published_goal(scen, world, rem[0])
    tp = world.tube_params
    tubes = [tp] if isinstance(tp, dict) else list(tp)
    succ = corridor_successors(scen, world)
    p = np.asarray(world.agents[i].state.p_pos, float)
    best = None
    for k, t in enumerate(tubes):
        if succ.get(k) is not None:
            continue
        ent, ex = np.asarray(t["entrance"], float), np.asarray(t["exit"], float)
        L = float(np.linalg.norm(ex - ent)); e = (ex - ent) / L; n = np.array([-e[1], e[0]])
        hw = float(t.get("half_width", float(t["width"]) / 2))
        r = p - ent; s_, y_ = float(r @ e), float(r @ n)
        if L < s_ <= L + gap_reach and abs(y_) <= hw + gap_reach and (best is None or s_ - L < best[0]):
            best = (s_ - L, k)
    return None if best is None else _published_goal(scen, world, best[1])


def _true_goal(scen, world, i):
    try:
        return np.asarray(scen.landmark_poses[scen.goal_match_index[world.agents[i].id]], dtype=float)
    except Exception:
        return None


def _remaining_tubes(scen, world, i):
    """Corridor indices aircraft i still has to fly, current first."""
    tp = world.tube_params
    if isinstance(tp, dict):
        return [0]
    n = len(tp)
    if hasattr(scen, "ROUTES") and hasattr(scen, "_route") and hasattr(scen, "_seg"):
        route = scen.ROUTES[int(scen._route[i])]
        return list(route[int(scen._seg[i]):])
    cur = int(scen.current_tube[i]) if getattr(scen, "current_tube", None) is not None else 0
    if n == 2:                       # sequential: 0 -> 1
        return list(range(cur, n))
    if cur in (0, 1):                # merge family: arm -> stem (-> turn)
        return [cur] + list(range(2, n))
    return list(range(cur, n))


def route_paths(scen, world, max_lateral=1.0, remaining=None, goal=None):
    """Per-aircraft route polyline (waypoints, km) for the "route" neighbour model:
    keep the aircraft's lateral offset (clamped to the corridor) through its current
    corridor to the exit, then follow the centre line of each remaining corridor.
    None for an aircraft well off its corridor (lateral offset beyond the half-width
    + max_lateral, or already past the exit of its last corridor)."""
    tp = world.tube_params
    tubes = [tp] if isinstance(tp, dict) else list(tp)
    out = RouteSet()
    goals, inside = [], []          # [ES] added: for the hybrid / goal variants
    for i, a in enumerate(world.agents):
        rem = None
        try:
            rem = (remaining or _remaining_tubes)(scen, world, i)
        except Exception:
            rem = None
        try:
            goals.append((goal or (lambda sc, w, k, r: _true_goal(sc, w, k)))(scen, world, i, rem or []))
        except Exception:
            goals.append(None)
        inside.append(False)
        try:
            if not rem:
                out.append(None)
                continue
            t0 = tubes[rem[0]]
            ent, ex = np.asarray(t0["entrance"], float), np.asarray(t0["exit"], float)
            e = (ex - ent) / np.linalg.norm(ex - ent)
            n = np.array([-e[1], e[0]])
            hw = float(t0.get("half_width", float(t0["width"]) / 2))
            r = np.asarray(a.state.p_pos, float) - ent
            s_, y_ = float(r @ e), float(r @ n)
            Lt = float(np.linalg.norm(ex - ent))
            inside[-1] = bool(0.0 <= s_ <= Lt and abs(y_) <= hw)
            if abs(y_) > hw + max_lateral or (len(rem) == 1 and s_ > Lt):
                out.append(None)
                continue
            yc = float(np.clip(y_, -0.8 * hw, 0.8 * hw))
            wp = []
            if s_ < 0:
                wp.append(ent + yc * n)
            wp.append(ex + yc * n)
            for k in rem[1:]:
                wp.append(np.asarray(tubes[k]["entrance"], float))
                wp.append(np.asarray(tubes[k]["exit"], float))
            out.append(np.array(wp))
        except Exception:
            out.append(None)
    out.goals, out.inside = goals, inside
    return out


# [ES] added: "route_map" -- the same route prediction, but each neighbour's route is
# inferred only from what the ego could sense (the neighbour's position and heading)
# plus the published corridor map (corridor geometry and which corridor follows
# which). Nothing is read from the simulator's record of the neighbour's corridor.
def corridor_successors(scen, world):
    """Published map: corridor index -> the corridor that follows it (None = last).
    Built from the layout definition only (static), never from aircraft state."""
    tp = world.tube_params
    if isinstance(tp, dict):
        return {0: None}
    n = len(tp)
    if hasattr(scen, "ROUTES"):
        succ = {k: None for k in range(n)}
        for route in scen.ROUTES:
            for a, b in zip(route[:-1], route[1:]):
                succ[a] = b
        return succ
    if n == 2:                          # sequential: 0 -> 1
        return {0: 1, 1: None}
    succ = {0: 2, 1: 2}                  # merge family: arms -> stem (-> turn)
    for k in range(2, n):
        succ[k] = k + 1 if k + 1 < n else None
    return succ


def _infer_remaining_tubes(scen, world, i, gap_reach=1.2, approach_lateral=1.0, approach_cos=0.5):
    """Corridors aircraft i still has to fly, inferred from its sensed position and
    heading and the published map only:
      1. inside a corridor (the one best aligned with its heading if several overlap,
         e.g. at a crossing) -> that corridor;
      2. else just past a corridor's exit (within gap_reach km, roughly in line) and
         that corridor has a successor -> the successor (it is in the junction gap);
      3. else behind a corridor's entrance, within approach_lateral km of its
         centre line extended backwards, and heading towards the entrance (within
         60 deg) -> that corridor, the nearest such entrance (queue / approach);
      4. else unknown -> [] (constant-velocity fallback)."""
    tp = world.tube_params
    tubes = [tp] if isinstance(tp, dict) else list(tp)
    succ = corridor_successors(scen, world)
    a = world.agents[i]
    p = np.asarray(a.state.p_pos, float)
    th = float(getattr(a.state, "theta", 0.0))
    hd = np.array([np.cos(th), np.sin(th)])
    inside, past, ahead = [], [], []
    for k, t in enumerate(tubes):
        ent, ex = np.asarray(t["entrance"], float), np.asarray(t["exit"], float)
        L = float(np.linalg.norm(ex - ent)); e = (ex - ent) / L; n = np.array([-e[1], e[0]])
        hw = float(t.get("half_width", float(t["width"]) / 2))
        r = p - ent; s_, y_ = float(r @ e), float(r @ n)
        if 0.0 <= s_ <= L and abs(y_) <= hw:
            inside.append((-float(hd @ e), k))
        elif L < s_ <= L + gap_reach and abs(y_) <= hw + gap_reach and succ.get(k) is not None:
            past.append((s_ - L, k))
        elif s_ < 0 and abs(y_) <= hw + approach_lateral:
            to_ent = ent - p
            d = float(np.linalg.norm(to_ent))
            if d < 1e-9 or float(hd @ to_ent) / d >= approach_cos:
                ahead.append((d, k))
    if inside:
        k = min(inside)[1]
    elif past:
        k = succ[min(past)[1]]
    elif ahead:
        k = min(ahead)[1]
    else:
        return []
    rem, seen = [k], {k}
    while succ.get(rem[-1]) is not None and succ[rem[-1]] not in seen:
        rem.append(succ[rem[-1]]); seen.add(rem[-1])
    return rem


def route_paths_map(scen, world, max_lateral=1.0):
    """route_paths with each aircraft's route inferred from sensing + the map."""
    return route_paths(scen, world, max_lateral=max_lateral, remaining=_infer_remaining_tubes, goal=_map_goal)
