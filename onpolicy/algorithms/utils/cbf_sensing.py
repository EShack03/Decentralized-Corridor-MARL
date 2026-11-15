"""[ES] added (13 Nov 2026): imperfect sensing of neighbours for the safety filter.

Robustness test only. The filter normally reads every neighbour's exact position,
heading and speed from the simulator. With this model it instead sees what a
real aircraft would receive from a broadcast surveillance system (ADS-B-like):
  * latency  -- each neighbour's state as it was `delay_steps` steps ago
               (1 step = 1 s), not extrapolated (uncompensated latency);
  * noise    -- independent Gaussian error on each neighbour's position
               (sigma per axis, km) and velocity vector (sigma per axis, km/s);
               heading and speed are recomputed from the noisy velocity.
The error is in the broadcast, so every receiving aircraft sees the same
erroneous state for a given neighbour at a given step. Each aircraft always
knows its OWN state exactly (the filter rolls the ego forward from its true
state). Route inference for the map-only / simulator-route models uses the same
sensed states. The policy's own observations are untouched -- only the safety
filter's neighbour information is degraded.

Noise uses its own random generator (seeded from --seed), so the environment's
random draws -- and therefore the episodes -- are unchanged. With all settings
at zero the model is never created and the filter is byte-for-byte as before.

Receiver-side mitigations (14 Nov 2026; both off by default):
  * smoothing     -- an alpha-beta-style tracker per neighbour on the received
                     stream: position x = x_pred + ALPHA (z - x_pred), with
                     x_pred = x_prev + v_prev dt; velocity vector
                     v = v_prev + BETA (v_meas - v_prev). Gains fixed before testing (see ALPHA/BETA).
  * compensation  -- broadcasts carry their time of applicability, so the
                     receiver knows each state's age d; it dead-reckons the
                     (smoothed) state forward along its velocity by d before the
                     filter uses it. Compensation needs the age only, never the
                     true state.
  * kalman        -- (15 Nov 2026) alternative tracker: a constant-velocity Kalman
                     filter per axis, measuring broadcast position AND velocity, with
                     the measurement noise set from the broadcast's stated accuracy
                     (ADS-B NACp / NACv; here the condition's sigmas) and white-
                     acceleration process noise SIGMA_A = 5 m/s^2 (chosen on recorded
                     flight paths from a separate run). Replaces the fixed-gain tracker.
Order: delayed, noisy broadcast -> tracker -> dead-reckoning by the age -> filter.
"""
import numpy as np


class SensingModel:
    RESET_JUMP_KM = 0.5   # any aircraft moving further than this in one step => new episode

    # Tracker gains, fixed before the mitigation test. Chosen on recorded sequential flight paths
    # (a separate run, not the robustness-test results) from a small grid {(0.4,0.3),(0.4,0.6),(0.5,0.8),
    # (0.3,0.8)}: (0.3, 0.8) had the lowest error for moving aircraft at 50 m noise (28 m vs 63 m raw),
    # 50 m + 1 s (30 vs 107 m) and 2 s delay (16 vs 185 m), at 7 m lag error with perfect sensing.
    ALPHA = 0.3   # position gain
    BETA = 0.8    # velocity gain (broadcast velocity is accurate relative to 90 m/s, so trust it)
    DT = 1.0      # s per step

    SIGMA_A = 0.005   # km/s^2 (5 m/s^2): Kalman process noise (manoeuvre acceleration); fixed before testing
    MIN_SIGMA = 1e-6  # km, km/s: floor on the stated accuracy (perfect sensing -> trust the broadcast)

    def __init__(self, pos_sigma_km=0.0, vel_sigma_kms=0.0, delay_steps=0, seed=0, smooth=False, compensate=False,
                 kalman=False, ground_vel=False):
        self.ground_vel = bool(ground_vel)   # [ES] added: broadcast the actual ground velocity (motion over the last second)
        self.prev_pos = None
        self.pos_sigma = float(pos_sigma_km)
        self.vel_sigma = float(vel_sigma_kms)
        self.delay = int(delay_steps)
        self.smooth = bool(smooth)
        self.compensate = bool(compensate)
        self.kalman = bool(kalman)
        self.track = None             # tracker state: (na, 4) [x, y, vx, vy], or (na, 2 axes, [p, v]) for Kalman
        self.kf_P = None              # (na, 2 axes, 2, 2) Kalman covariance
        self.rng = np.random.default_rng(1_000_003 + int(seed))
        self.hist = []
        self.stats = {"steps": 0, "resets": 0, "sum_pos_err": 0.0, "n": 0}

    @property
    def active(self):
        return (self.pos_sigma > 0 or self.vel_sigma > 0 or self.delay > 0 or self.smooth or self.compensate
                or self.kalman or self.ground_vel)

    def reset(self):
        self.prev_pos = None
        self.hist = []
        self.track = None
        self.kf_P = None
        self.stats["resets"] += 1

    def apply(self, wstate, as_float32=True, parked=None):
        """wstate: (na, 4) true [x, y, theta, v] this step -> (na, 4) sensed.
        parked: optional (na,) bool -- parked aircraft broadcast zero velocity. (The simulator freezes a
        parked aircraft's position but leaves its speed state unchanged; a real parked aircraft reports 0.)"""
        w = np.asarray(wstate, dtype=np.float64)
        if parked is not None and np.any(parked):
            w = w.copy(); w[np.asarray(parked, dtype=bool), 3] = 0.0
        if self.hist and np.max(np.linalg.norm(w[:, :2] - self.hist[-1][:, :2], axis=1)) > self.RESET_JUMP_KM:
            self.reset()
        if self.ground_vel:
            # GNSS-like velocity: the aircraft's actual displacement over the last second (track angle and
            # ground speed). The simulator's speed state is capped after each step, so at full speed it
            # understates the motion by 3-5 %. First step of an episode: the state (no previous position).
            pos_now = w[:, :2].copy()
            if self.prev_pos is not None and self.prev_pos.shape == pos_now.shape:
                d = (pos_now - self.prev_pos) / self.DT
                gs = np.linalg.norm(d, axis=1)
                w = w.copy()
                mv = gs > 1e-9
                w[mv, 2] = np.arctan2(d[mv, 1], d[mv, 0])
                w[:, 3] = gs
            self.prev_pos = pos_now
        self.hist.append(w.copy())
        if len(self.hist) > self.delay + 1:
            self.hist = self.hist[-(self.delay + 1):]
        src = self.hist[max(0, len(self.hist) - 1 - self.delay)]
        out = src.copy()
        na = out.shape[0]
        if self.pos_sigma > 0:
            out[:, :2] += self.rng.normal(0.0, self.pos_sigma, size=(na, 2))
        if self.vel_sigma > 0:
            vel = out[:, 3:4] * np.stack([np.cos(out[:, 2]), np.sin(out[:, 2])], axis=1)
            vel += self.rng.normal(0.0, self.vel_sigma, size=(na, 2))
            out[:, 2] = np.arctan2(vel[:, 1], vel[:, 0])
            out[:, 3] = np.linalg.norm(vel, axis=1)
        if self.smooth or self.compensate or self.kalman:
            out = self._receiver(out)
        self.stats["steps"] += 1
        self.stats["sum_pos_err"] += float(np.sum(np.linalg.norm(out[:, :2] - w[:, :2], axis=1)))
        self.stats["n"] += na
        return out.astype(np.float32) if as_float32 else out

    def _receiver(self, meas):
        """Receiver-side processing of the (delayed, noisy) broadcasts: optional tracker, then
        optional dead-reckoning by the known age of the data."""
        z = meas[:, :2]
        vz = meas[:, 3:4] * np.stack([np.cos(meas[:, 2]), np.sin(meas[:, 2])], axis=1)
        if self.kalman:
            pos, vel = self._kalman(z, vz)
        elif self.smooth:
            if self.track is None:
                self.track = np.concatenate([z, vz], axis=1)
            else:
                x_pred = self.track[:, :2] + self.track[:, 2:] * self.DT
                pos = x_pred + self.ALPHA * (z - x_pred)
                vel = self.track[:, 2:] + self.BETA * (vz - self.track[:, 2:])
                self.track = np.concatenate([pos, vel], axis=1)
            pos, vel = self.track[:, :2].copy(), self.track[:, 2:].copy()
        else:
            pos, vel = z.copy(), vz.copy()
        if self.compensate and self.delay > 0:
            age = min(self.delay, len(self.hist) - 1) * self.DT   # the age actually used this step
            pos = pos + vel * age
        out = meas.copy()
        out[:, :2] = pos
        out[:, 2] = np.arctan2(vel[:, 1], vel[:, 0])
        out[:, 3] = np.linalg.norm(vel, axis=1)
        return out

    def _kalman(self, z, vz):
        """Constant-velocity Kalman filter, x and y axes independent. State per axis [p, v];
        measurement per axis [p, v] (ADS-B broadcasts both). R from the stated accuracy, Q from SIGMA_A."""
        F = np.array([[1.0, self.DT], [0.0, 1.0]])
        G = np.array([[0.5 * self.DT ** 2], [self.DT]])
        Q = (self.SIGMA_A ** 2) * (G @ G.T)
        R = np.diag([max(self.pos_sigma, self.MIN_SIGMA) ** 2, max(self.vel_sigma, self.MIN_SIGMA) ** 2])
        meas = np.stack([np.stack([z[:, 0], vz[:, 0]], 1), np.stack([z[:, 1], vz[:, 1]], 1)], 1)   # (na, axis, [p, v])
        if self.track is None or self.kf_P is None:
            self.track = meas.copy()
            self.kf_P = np.broadcast_to(R, meas.shape[:2] + (2, 2)).copy()
        else:
            X = self.track @ F.T                               # predict
            P = F @ self.kf_P @ F.T + Q
            K = P @ np.linalg.inv(P + R)                       # gain (measurement matrix = identity)
            self.track = X + (K @ (meas - X)[..., None])[..., 0]
            self.kf_P = (np.eye(2) - K) @ P
        pos = self.track[:, :, 0].copy(); vel = self.track[:, :, 1].copy()
        return pos, vel

    def summary(self):
        n = max(self.stats["n"], 1)
        return (f"sensing model: position sigma {self.pos_sigma * 1000:.0f} m, velocity sigma "
                f"{self.vel_sigma * 1000:.1f} m/s, delay {self.delay} s, smoothing {'on' if self.smooth else 'off'}, "
                f"Kalman {'on' if self.kalman else 'off'}, "
                f"latency compensation {'on' if self.compensate else 'off'}; mean position error seen by "
                f"the filter {1000 * self.stats['sum_pos_err'] / n:.1f} m over {self.stats['steps']} steps, "
                f"{self.stats['resets']} episode resets")


class _SensedState:
    def __init__(self, s, orig):
        self.p_pos = np.array(s[:2], dtype=float)
        self.theta = float(s[2])
        self.speed = float(s[3])
        self._orig = orig

    def __getattr__(self, k):
        return getattr(self._orig, k)


class _SensedAgent:
    def __init__(self, s, orig):
        self.state = _SensedState(s, orig.state)
        self._orig = orig

    def __getattr__(self, k):
        return getattr(self._orig, k)


class SensedWorld:
    """Read-only view of the world in which every agent's position, heading and speed
    are the sensed values; everything else (corridors, landmarks, ...) is the real world.
    Used only for route inference."""

    def __init__(self, world, sensed):
        self._world = world
        self.agents = [_SensedAgent(s, a) for s, a in zip(sensed, world.agents)]

    def __getattr__(self, k):
        return getattr(self._world, k)


# ---------------------------------------------------------------------------------------------
# [ES] added (14 Nov 2026): the same imperfect sensing for the POLICY's observations of its
# neighbours (one surveillance picture shared by policy and filter). Attached to an environment
# instance by wrapping its observation callbacks -- no scenario or environment file is edited.
# While an aircraft's observation is built, every OTHER aircraft's position, velocity and heading
# are swapped for their sensed values (and the graph distances recomputed from them); the
# aircraft's own state stays exact; everything is restored immediately afterwards, so the
# simulation itself (dynamics, rewards, success, losses) always uses the true states.
# ---------------------------------------------------------------------------------------------
def _true_states(world):
    """(na, 4) [x, y, theta, v] -- the air-taxi state vector itself."""
    return np.array([np.asarray(a.state.values[:4], dtype=np.float64) for a in world.agents])


class PolicySensing:
    def __init__(self, env, model):
        self.env, self.model = env, model
        self.world = env.world
        self._key = None
        self.sensed = None
        self._obs = env.observation_callback
        self._gobs = env.graph_observation_callback
        env.observation_callback = self._observation
        if self._gobs is not None:
            env.graph_observation_callback = self._graph_observation

    def _refresh(self):
        w = _true_states(self.world)
        key = w[:, :2].tobytes()
        if key != self._key:                      # once per simulator step (or reset)
            self._key = key
            parked = np.array([bool(getattr(a, "status", False)) for a in self.world.agents])
            self.sensed = self.model.apply(w, as_float32=False, parked=parked)
            self.world._sensed_state = self.sensed.astype(np.float32)   # read by the filter hook
        return self.sensed

    def _swap(self, ego):
        s = self._refresh(); saved = []
        for i, a in enumerate(self.world.agents):
            if a is ego:
                continue
            st = a.state
            saved.append((st, st.values))
            v = np.array(st.values, dtype=np.float64, copy=True)
            v[:4] = s[i, :4]
            st.values = v
        return saved

    @staticmethod
    def _restore(saved):
        for st, vals in saved:
            st.values = vals

    def _observation(self, agent, world):
        saved = self._swap(agent)
        try:
            return self._obs(agent=agent, world=world)
        finally:
            self._restore(saved)

    def _graph_observation(self, agent, world):
        saved = self._swap(agent)
        cached = world.cached_dist_mag
        try:
            if cached is not None:
                pos = np.array([e.state.p_pos for e in world.entities], dtype=np.float64)
                world.cached_dist_mag = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
            return self._gobs(agent, world)
        finally:
            world.cached_dist_mag = cached
            self._restore(saved)


def attach_policy_sensing(env, model):
    """Wrap one GraphMPE environment instance (e.g. envs.envs[0]) so its policy observations use `model`."""
    return PolicySensing(env, model)
