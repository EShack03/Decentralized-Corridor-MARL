"""
CBF evaluation metrics — the rich safety/severity statistics layer for the
in-loop evaluator.

This module exists so the ORIGINAL eval loop (GMPERunner.render) can stay almost
untouched: the runner creates one CBFEvalCollector and drives it with a handful
of guarded one-line calls, while ALL the heavy logic (per-step separation
geometry, the in/outside-corridor split, the summary table, the result-file
naming convention) lives here.

Nothing in here affects the policy, the environment or the RNG. Every method is a
read-only observation of world state plus bookkeeping, so with the collector
present the original's own metrics (collisions, success, conformance) are
byte-for-byte unchanged — only EXTRA numbers are produced alongside them.

Metric definitions (consistent with the project's metric-caveats doc):
  * collisions / success / conformance / spacing come from the ORIGINAL runner
    helpers (get_collisions / get_fraction_episodes / get_conformation_percentages
    / get_spacing_violations) and are handed to the collector each episode; the
    collector only reports them, it does not recompute them.
  * separation severity (min sep, closest approach, max intrusion depth,
    integrated intrusion, near-miss rate, collision-free-episode %) and the
    in/outside-corridor split are computed here from post-step world positions,
    over the SAME active-aircraft set the env uses for collisions (a parked
    aircraft on its goal is not a traffic conflict).
  * throughput / completion time are the ORIGINAL loop's own values (env real
    timebase), passed in per episode.
"""

import datetime
import numpy as np


# --- air_taxi constants (mirrored from multiagent/config.AirTaxiConfig) -------
DT = 1.0
SEP = 500 * 0.0003048              # separation standard = 0.1524 km (500 ft)
NEARMISS_MULT = 2.0                # proximity-event buffer = NEARMISS_MULT * SEP


# --- scenario corridor query --------------------------------------------------
def get_scenario(envs):
    """The scenario instance behind the (single) env, via its bound info_callback.
    Used only for the side-effect-free is_in_tube() query; None if unavailable."""
    try:
        return envs.envs[0].info_callback.__self__
    except Exception:
        return None


def in_corridor_flags(scen, world, world_agents):
    """Per-agent bool: is this agent inside its own corridor tube right now?
    Uses the scenario's pure is_in_tube(world, pos, current_tube). None if
    unavailable, so callers fall back to unsplit (overall-only) severity."""
    if scen is None or not hasattr(scen, "is_in_tube"):
        return None
    try:
        return np.array(
            [bool(scen.is_in_tube(world, a.state.p_pos, int(scen.current_tube[a.id])))
             for a in world_agents],
            dtype=bool)
    except Exception:
        return None


# --- per-episode separation / severity accumulator ----------------------------
class _SepAccum:
    """Accumulates pairwise separation / severity over one episode, split into
    'in-corridor' and 'outside' by whether EITHER agent of a conflicting pair is
    inside its tube. Every pair is visited once (strict upper triangle), so there
    is no double-counting and in + out == total for every quantity."""

    def __init__(self):
        self.closest = {"all": np.inf, "in": np.inf, "out": np.inf}
        self.intrusion = {"all": 0.0, "in": 0.0, "out": 0.0}   # sum max(0, SEP-d)
        self.maxintr = {"all": 0.0, "in": 0.0, "out": 0.0}     # worst single-step depth
        self.los = {"all": 0, "in": 0, "out": 0}               # pair-steps d < SEP
        self.near_pairsteps = 0
        self.pairsteps = 0

    def update(self, pos, in_corr, active=None):
        """pos: (na, 2) post-step positions. in_corr: (na,) bool or None (no split).
        active: (na,) bool or None. A pair is scored ONLY if both aircraft are
        active (not parked at goal), matching the env's is_collision guard, so the
        severity numbers reconcile with the env's own collision counter."""
        if pos.shape[0] < 2:
            return
        i, j = np.triu_indices(pos.shape[0], k=1)
        d = np.linalg.norm(pos[i] - pos[j], axis=1)
        if active is not None:
            keep = active[i] & active[j]
            if not keep.any():
                return
            i, j, d = i[keep], j[keep], d[keep]
        depth = np.maximum(0.0, SEP - d)
        los = d < SEP
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
            self.intrusion[key] += float(depthsel.sum())
            self.maxintr[key] = max(self.maxintr[key], float(depthsel.max()))
            self.los[key] += int(lossel.sum())
        self.near_pairsteps += int((d < NEARMISS_MULT * SEP).sum())
        self.pairsteps += d.shape[0]


# --- result-file naming convention --------------------------------------------
# <stage>_<policy>_<filter>_<scenario><N>_<episodes>ep_<seeds>_<YYMMDD>.txt
_SCENARIO_SHORT = {
    "working_three_phase_graph": "single",
    "three_phase_graph_sequential": "seq",
    "three_phase_graph_merge": "merge",
    "three_phase_graph_split": "split",
}


def result_filename(all_args, filtered):
    stage = getattr(all_args, "run_stage", None) or getattr(all_args, "eval_tag", "eval")
    policy = getattr(all_args, "policy_slug", None) or getattr(all_args, "eval_tag", "eval")
    scen = _SCENARIO_SHORT.get(all_args.scenario_name, all_args.scenario_name)
    N = all_args.num_agents
    eps = all_args.render_episodes
    seeds = list(getattr(all_args, "_seeds", [int(all_args.seed)]))
    seedslug = f"{len(seeds)}seed" if len(seeds) > 1 else f"seed{int(all_args.seed)}"
    filt = "cv" if filtered and getattr(all_args, "cbf_neighbour_model", "static") == "constant_velocity" \
        else ("static" if filtered else "none")
    date = datetime.datetime.now().strftime("%y%m%d")
    return f"{stage}_{policy}_{filt}_{scen}N{N}_{eps}ep_{seedslug}_{date}.txt"


# --- the collector ------------------------------------------------------------
class CBFEvalCollector:
    """Drives the extra metrics from inside render() with minimal footprint.

    Lifecycle per render():
        begin_episode()                         # once at the top of each episode
        step(world_agents, scen, world)         # once after each envs.step()
        filter_step(no_safe, n_blocked, n_agents)  # once per step IF filtering
        end_episode(num_collisions, success, conf_viol, spacing_viol,
                    throughput, completion_time)   # once at episode end
        finalize(all_args) -> prints summary + saves result file, returns dict
    """

    def __init__(self, filtered):
        self.filtered = bool(filtered)
        self._sep = None
        self.coll, self.succ, self.conf_viol = [], [], []
        self.spacing, self.through, self.comp = [], [], []
        B = ("all", "in", "out")
        self.closest = {b: [] for b in B}
        self.intrusion = {b: [] for b in B}
        self.maxintr = {b: [] for b in B}
        self.los = {b: [] for b in B}
        self.nearmiss_rate, self.cfree = [], []
        self.have_split = False
        self.n_steps = 0
        self.n_no_safe = 0
        self.n_blocked = 0

    def begin_episode(self):
        self._sep = _SepAccum()

    def step(self, world_agents, scen, world):
        pos = np.array([a.state.p_pos for a in world_agents], dtype=np.float64)
        active = np.array([not bool(getattr(a, "status", False))
                           for a in world_agents], dtype=bool)
        in_corr = in_corridor_flags(scen, world, world_agents)
        if in_corr is not None:
            self.have_split = True
        self._sep.update(pos, in_corr, active)

    def filter_step(self, no_safe_count, n_blocked, n_agents):
        self.n_steps += n_agents
        self.n_no_safe += int(no_safe_count)
        self.n_blocked += int(n_blocked)

    def end_episode(self, num_collisions, success, conf_viol, spacing_viol,
                    throughput, completion_time):
        self.coll.append(float(num_collisions))
        self.succ.append(float(np.mean(success)))
        self.conf_viol.append(float(np.mean(conf_viol)) if len(np.atleast_1d(conf_viol)) else 0.0)
        self.spacing.append(float(np.mean(spacing_viol)) if len(np.atleast_1d(spacing_viol)) else 0.0)
        self.through.append(float(throughput))
        self.comp.append(float(completion_time))
        s = self._sep
        for b in ("all", "in", "out"):
            c = s.closest[b]
            self.closest[b].append(c if np.isfinite(c) else np.nan)
            self.intrusion[b].append(s.intrusion[b])
            self.maxintr[b].append(s.maxintr[b])
            self.los[b].append(s.los[b])
        self.nearmiss_rate.append(s.near_pairsteps / s.pairsteps if s.pairsteps else 0.0)
        self.cfree.append(1.0 if s.los["all"] == 0 else 0.0)

    def summary(self):
        def m(x):
            x = np.array(x, dtype=float)
            return float(np.nanmean(x)) if len(x) else float("nan")
        out = {
            "collisions": m(self.coll),
            "success_pct": 100.0 * m(self.succ),
            "conformance_pct": 100.0 * (1.0 - m(self.conf_viol)),
            "spacing_violation_pct": 100.0 * m(self.spacing),
            "throughput": m(self.through),
            "completion_time_s": m(self.comp),
            "min_sep_km": float(np.nanmin(self.closest["all"])) if len(self.closest["all"]) else float("nan"),
            "mean_closest_km": m(self.closest["all"]),
            "max_intrusion_km": m(self.maxintr["all"]),
            "integrated_intrusion_km": m(self.intrusion["all"]),
            "nearmiss_rate_pct": 100.0 * m(self.nearmiss_rate),
            "collision_free_pct": 100.0 * m(self.cfree),
            "los_pairsteps": m(self.los["all"]),
        }
        for b in ("in", "out"):
            out[f"los_pairsteps_{b}"] = m(self.los[b])
            out[f"max_intrusion_km_{b}"] = m(self.maxintr[b])
            cb = np.array(self.closest[b], dtype=float)
            out[f"min_sep_km_{b}"] = float(np.nanmin(cb)) if np.isfinite(cb).any() else float("nan")
        if self.filtered and self.n_steps:
            out["no_safe_action_rate"] = self.n_no_safe / self.n_steps
            out["avg_primitives_blocked"] = self.n_blocked / self.n_steps
            out["agent_steps"] = self.n_steps
        return out

    def _lines(self, all_args, s):
        kind = "FILTERED SYSTEM" if self.filtered else "BASELINE (no filter)"
        L = []
        L.append("=" * 62)
        L.append(f"CBF EVAL RESULT [{getattr(all_args,'eval_tag','eval')}] - {kind}  "
                 f"(N={all_args.num_agents}, episodes={all_args.render_episodes}, "
                 f"seed={all_args.seed})")
        if self.filtered:
            L.append(f"  filter: neighbour={getattr(all_args,'cbf_neighbour_model','static')}, "
                     f"alpha={getattr(all_args,'cbf_alpha',1.0)}, "
                     f"horizon={getattr(all_args,'cbf_horizon',3)}, "
                     f"fallback={getattr(all_args,'cbf_fallback','least_unsafe')}")
        L.append("=" * 62)
        L.append(f"{'metric':<30}{'value':>16}")
        L.append(f"{'collisions (LoS)':<30}{s['collisions']:>16.3f}")
        L.append(f"{'  collisions cross-check':<30}{s['los_pairsteps']:>16.3f}")
        L.append(f"{'mean success rate':<30}{s['success_pct']:>15.2f}%")
        L.append(f"{'corridor conformance':<30}{s['conformance_pct']:>15.2f}%")
        L.append(f"{'collision-free episodes':<30}{s['collision_free_pct']:>15.1f}%")
        L.append(f"{'min separation (km)':<30}{s['min_sep_km']:>16.3f}")
        L.append(f"{'mean closest approach (km)':<30}{s['mean_closest_km']:>16.3f}")
        L.append(f"{'max intrusion depth (km)':<30}{s['max_intrusion_km']:>16.3f}")
        L.append(f"{'integrated intrusion (km.st)':<30}{s['integrated_intrusion_km']:>16.3f}")
        L.append(f"{'near-miss rate %':<30}{s['nearmiss_rate_pct']:>16.2f}")
        L.append(f"{'spacing violations %':<30}{s['spacing_violation_pct']:>16.2f}")
        L.append(f"{'throughput (agents/min)':<30}{s['throughput']:>16.3f}")
        L.append(f"{'completion time (s)':<30}{s['completion_time_s']:>16.2f}")
        if self.have_split:
            L.append("")
            L.append("IN- vs OUTSIDE-CORRIDOR  (in-corridor = either aircraft inside its tube;"
                     " in + out = total)")
            L.append(f"{'collisions in corridor':<30}{s['los_pairsteps_in']:>16.3f}")
            L.append(f"{'collisions outside corridor':<30}{s['los_pairsteps_out']:>16.3f}")
            L.append(f"{'max intrusion in (km)':<30}{s['max_intrusion_km_in']:>16.3f}")
            L.append(f"{'max intrusion out (km)':<30}{s['max_intrusion_km_out']:>16.3f}")
        if self.filtered and "agent_steps" in s:
            L.append("")
            L.append(f"no-safe-action rate    : {s['no_safe_action_rate']*100:.2f}%  "
                     f"(of {s['agent_steps']} agent-steps)")
            L.append(f"avg primitives blocked : {s['avg_primitives_blocked']:.2f} / 25 per step")
        return L

    def finalize(self, all_args):
        s = self.summary()
        for ln in self._lines(all_args, s):
            print(ln)
        out_path = result_filename(all_args, self.filtered)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# CBF eval result [{getattr(all_args,'eval_tag','eval')}]  "
                    f"{datetime.datetime.now()}\n")
            f.write(f"# {'FILTERED SYSTEM' if self.filtered else 'RAW BASELINE (no filter)'}  "
                    f"(computed inside the original render() loop)\n")
            f.write(f"# N={all_args.num_agents}  episodes={all_args.render_episodes}  "
                    f"seed={all_args.seed}  scenario={all_args.scenario_name}\n")
            f.write(f"# collisions = losses of separation (LoS): agent-pair-steps within "
                    f"{SEP:.4f} km (500 ft), per episode\n\n")
            for ln in self._lines(all_args, s):
                f.write(ln + "\n")
        print(f"\nsaved result -> {out_path}")

        # --- per-episode CSV (added) -------------------------------------------
        # One row per episode. The .txt above holds only this run's MEANS, which
        # cannot be pooled into a correct standard error. This CSV holds the raw
        # per-episode values, so several runs of the SAME config (e.g. parallel
        # seeds) can be concatenated and treated as one larger run: a proper
        # standard error over all episodes, percentiles, and a histogram. Writing
        # it never touches the simulation — it is a dump of already-collected lists.
        import csv as _csv
        ep_path = (out_path[:-4] if out_path.endswith(".txt") else out_path) + "_perep.csv"
        n_ep = len(self.coll)
        with open(ep_path, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["episode", "collisions", "success_pct", "conformance_pct",
                        "collision_free", "min_sep_km", "max_intrusion_km",
                        "integrated_intrusion", "nearmiss_pct", "spacing_pct",
                        "throughput", "completion_s", "los_pairsteps"])
            for i in range(n_ep):
                w.writerow([
                    i + 1,
                    self.coll[i],
                    100.0 * self.succ[i],
                    100.0 * (1.0 - self.conf_viol[i]),
                    int(self.cfree[i]),
                    self.closest["all"][i],
                    self.maxintr["all"][i],
                    self.intrusion["all"][i],
                    100.0 * self.nearmiss_rate[i],
                    100.0 * self.spacing[i],
                    self.through[i],
                    self.comp[i],
                    self.los["all"][i],
                ])
        print(f"saved per-episode -> {ep_path}")
        return s
