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

    def __init__(self, pos_sigma_km=0.0, vel_sigma_kms=0.0, delay_steps=0, seed=0, smooth=False, compensate=False):
        self.pos_sigma = float(pos_sigma_km)
        self.vel_sigma = float(vel_sigma_kms)
        self.delay = int(delay_steps)
        self.smooth = bool(smooth)
        self.compensate = bool(compensate)
        self.track = None             # (na, 4) [x, y, vx, vy] tracker state
        self.rng = np.random.default_rng(1_000_003 + int(seed))
        self.hist = []
        self.stats = {"steps": 0, "resets": 0, "sum_pos_err": 0.0, "n": 0}

    @property
    def active(self):
        return self.pos_sigma > 0 or self.vel_sigma > 0 or self.delay > 0 or self.smooth or self.compensate

    def reset(self):
        self.hist = []
        self.track = None
        self.stats["resets"] += 1

    def apply(self, wstate):
        """wstate: (na, 4) true [x, y, theta, v] this step -> (na, 4) sensed."""
        w = np.asarray(wstate, dtype=np.float64)
        if self.hist and np.max(np.linalg.norm(w[:, :2] - self.hist[-1][:, :2], axis=1)) > self.RESET_JUMP_KM:
            self.reset()
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
        if self.smooth or self.compensate:
            out = self._receiver(out)
        self.stats["steps"] += 1
        self.stats["sum_pos_err"] += float(np.sum(np.linalg.norm(out[:, :2] - w[:, :2], axis=1)))
        self.stats["n"] += na
        return out.astype(np.float32)

    def _receiver(self, meas):
        """Receiver-side processing of the (delayed, noisy) broadcasts: optional tracker, then
        optional dead-reckoning by the known age of the data."""
        z = meas[:, :2]
        vz = meas[:, 3:4] * np.stack([np.cos(meas[:, 2]), np.sin(meas[:, 2])], axis=1)
        if self.smooth:
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

    def summary(self):
        n = max(self.stats["n"], 1)
        return (f"sensing model: position sigma {self.pos_sigma * 1000:.0f} m, velocity sigma "
                f"{self.vel_sigma * 1000:.1f} m/s, delay {self.delay} s, smoothing {'on' if self.smooth else 'off'}, "
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
