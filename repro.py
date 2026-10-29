#!/usr/bin/env python3
# ============================================================
# repro.py — cross-platform reproduction driver (Windows + Linux)
# ------------------------------------------------------------
# One tool, four subcommands. No bash / grep needed; only Python + uv + git.
#
#   python repro.py setup      # build the env with uv (auto-detects GPU stack)
#   python repro.py patch      # apply the two quality patches (idempotent)
#   python repro.py check      # GPU / weights / minimal-eval self-check
#   python repro.py eval       # reproduce all evaluable scenarios -> results txt
#
# Run every command from the ROOT of the cloned repo (where onpolicy/ lives).
# Prereqs: `git clone` already done + `uv` installed (https://astral.sh/uv).
# ============================================================
import argparse
import datetime
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_MARKER = Path("onpolicy/scripts/eval_mpe.py")

# runtime deps for the Blackwell path (versions matching the validated env)
RUNTIME_DEPS = [
    "numpy==1.26.4", "gym==0.26.2", "gymnasium==1.3.0", "pyglet==1.5.26",
    "networkx==3.6.1", "scipy==1.13.1", "wandb==0.28.0",
    "tensorboard==2.17.1", "tensorboardX==2.6.1",
    "setproctitle==1.3.7", "absl-py==2.4.0", "imageio==2.31.1",
    "tqdm==4.66.2", "PyYAML==6.0.3",
]

# (tag, scenario_name, num_agents, world_size, episode_length)
SCENARIOS = [
# --- Sequential (the problem scenario) — full density sweep ---
    #("seq_N2",  "three_phase_graph_sequential",  2,  5, 200),
    #("seq_N3",  "three_phase_graph_sequential",  3,  5, 200),
    #("seq_N4",  "three_phase_graph_sequential",  4,  5, 200),
    #("seq_N5",  "three_phase_graph_sequential",  5,  5, 200),
    #("seq_N6",  "three_phase_graph_sequential",  6,  5, 200),
    #("seq_N8",  "three_phase_graph_sequential",  8,  5, 200),
    ("seq_N10", "three_phase_graph_sequential",  10, 5, 200),
# --- Single (control baseline) — matched densities ---
    #("single_N2",  "working_three_phase_graph",  2,  5, 120),
    #("single_N3",  "working_three_phase_graph",  3,  5, 120),
    #("single_N4",  "working_three_phase_graph",  4,  5, 120),
    #("single_N5",  "working_three_phase_graph",  5,  5, 120),
    #("single_N6",  "working_three_phase_graph",  6,  5, 120),
    #("single_N8",  "working_three_phase_graph",  8,  5, 120),
    #("single_N10", "working_three_phase_graph",  10, 5, 120),
# --- Merge (currently unused) scenarios ---
    #("merge_N3",   "three_phase_graph_merge",       3,  5, 200),
    #("merge_N5",   "three_phase_graph_merge",       5,  5, 200),
    #("merge_N10",  "three_phase_graph_merge",       10, 5, 200),
]

METRIC_KEYS = ("Success rates mean", "Conformance C% Mean",
               "Total Time Taken Median", "Num collisions",
               "Spacing violations Mean", "Delta_space Mean")


# ============================================================================
# CBF safety-layer evaluation (added) — tables used by the `reproduce` subcommand.
#
# Every CBF run drives the SAME eval_mpe.py -> render() loop as `eval` above, so
# the original metrics (collisions/success/conformance) are identical by
# construction. It adds the read-only severity metrics in every run, and the
# CBF safety filter only when a filtered configuration is chosen. A no-filter
# run of the original policy reproduces the `eval` numbers.
# ============================================================================
CBF_STAGES = {
    # stage       weights folder                                   policy slug      default filter
    "original": dict(tag="s0",  policy="original",
                     weights="model_weights", filt="none"),
    "s2":       dict(tag="s2",  policy="original",
                     weights="model_weights", filt="cv"),
    "s3A":      dict(tag="s3A", policy="shielded-train",
                     weights=os.path.join("onpolicy", "results", "GraphMPE",
                         "three_phase_graph_updated", "rmappo",
                         "stage3a_full", "run2", "models"), filt="cv"),
    "s3B":      dict(tag="s3B", policy="shaped-train",
                     weights=os.path.join("onpolicy", "results", "GraphMPE",
                         "working_three_phase_graph", "rmappo",
                         "stage3b_full_v2", "run1", "models"), filt="cv"),
}
CBF_SCENARIOS = {
    "single":     "working_three_phase_graph",
    "sequential": "three_phase_graph_sequential",
    "merge":      "three_phase_graph_merge",
    "combined":   "three_phase_graph_split_and_merge",
    "combinedfull": "three_phase_graph_combined_splits_and_merges_and_sequential",
    "mergeseq":   "three_phase_graph_merge_sequential",
    "mergeseqhard": "three_phase_graph_merge_sequential_hard",
}
CBF_EPLEN = {"single": 120, "sequential": 200, "merge": 200, "combined": 200,
             "combinedfull": 200, "mergeseq": 200, "mergeseqhard": 200}
CBF_FILTER_NEIGHBOUR = {"static": "static", "cv": "constant_velocity"}
CBF_BARRIER_DEFAULT = "cbf_stage1_barrier.pt"


# ---------- helpers ----------
def die(msg):
    print("ERROR:", msg, file=sys.stderr)
    sys.exit(1)


def ensure_repo_root():
    if not REPO_MARKER.exists():
        die("run from the repo root (onpolicy/scripts/eval_mpe.py not found)")


def ensure_uv():
    if shutil.which("uv") is None:
        die("uv not installed. Install:  https://astral.sh/uv  "
            "(curl -LsSf https://astral.sh/uv/install.sh | sh)")


def venv_python() -> Path:
    """Location of the venv interpreter differs by OS."""
    if os.name == "nt":
        return Path(".venv") / "Scripts" / "python.exe"
    return Path(".venv") / "bin" / "python"


def need_venv() -> str:
    py = venv_python()
    if not py.exists():
        die(f"venv not found at {py} -- run  python repro.py setup  first")
    return str(py)


def run(cmd, **kw):
    print(">>>", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def gpu_compute_cap():
    """Return the first GPU's compute capability as a float, or None."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, check=True).stdout
        for line in out.splitlines():
            line = line.strip()
            if line:
                return float(line)
    except Exception:
        return None
    return None


def verify_gpu(py):
    run([py, "-c",
         "import torch; assert torch.cuda.is_available(), 'CUDA not available'; "
         "print('torch', torch.__version__, '| device', torch.cuda.get_device_name(), "
         "'| cap', torch.cuda.get_device_capability())"])


def eval_cmd(py, tag, scenario, n, world, eplen, episodes, ifi="0.0", seed="1"):
    return [
        py, "-u", "onpolicy/scripts/eval_mpe.py", "--model_dir", "model_weights",
        "--scenario_name", scenario, "--dynamics_type", "air_taxi",
        "--num_agents", n, "--num_landmarks", n,
        "--world_size", world, "--episode_length", eplen,
        "--formation_type", "point", "--total_actions", "9", "--zeroshift", "10",
        "--render_episodes", episodes, "--use_dones", "False",
        "--ifi", ifi, "--model_name", tag, "--seed", str(seed),
    ]


# ---------- subcommand: setup ----------
def cmd_setup(args):
    ensure_repo_root()
    ensure_uv()

    stack = args.stack
    if stack == "auto":
        cap = gpu_compute_cap()
        if cap is None:
            die("could not detect GPU compute capability via nvidia-smi; "
                "pass --stack blackwell  or  --stack default  explicitly")
        stack = "blackwell" if cap >= 12.0 else "default"
        print(f">>> detected compute capability {cap} -> stack: {stack}")

    print(">>> [1/N] Create .venv (Python 3.11)")
    run(["uv", "venv", "--python", "3.11", ".venv"])
    py = str(venv_python())

    if stack == "blackwell":
        print(">>> Install torch 2.11.0 (cu128, supports Blackwell sm_120)")
        run(["uv", "pip", "install", "--python", py, "torch==2.11.0",
             "--index-url", "https://download.pytorch.org/whl/cu128"])
        print(">>> Install PyG compiled deps (prebuilt wheels, pt2.11+cu128)")
        run(["uv", "pip", "install", "--python", py,
             "torch-scatter==2.1.2", "torch-sparse==0.6.18",
             "-f", "https://data.pyg.org/whl/torch-2.11.0+cu128.html"])
        run(["uv", "pip", "install", "--python", py, "torch-geometric==2.3.1"])
        print(">>> Install remaining runtime deps")
        run(["uv", "pip", "install", "--python", py, *RUNTIME_DEPS])
    else:  # default (pre-Blackwell): repo's pinned torch 2.0.1 stack via uv
        if not Path("requirements.txt").exists():
            die("requirements.txt not found (needed for the default stack)")
        print(">>> Install repo requirements (torch 2.0.1) + PyG wheel index")
        run(["uv", "pip", "install", "--python", py,
             "-r", "requirements.txt", "-f", args.pyg_index])
        print(">>> Add gymnasium (imported by the code but missing from requirements.txt)")
        run(["uv", "pip", "install", "--python", py, "gymnasium==1.3.0"])

    print("\n>>> Verify GPU visibility")
    verify_gpu(py)
    print("\nOK: environment ready. Next:  python repro.py patch")


# ---------- subcommand: patch ----------
def _patch_file(path, old, new, marker):
    s = open(path, encoding="utf-8").read()  # text mode normalizes CRLF/LF -> \n
    if marker in s:
        print(f"  {path}: patch already present, skipping")
        return
    if old not in s:
        die(f"patch target not found in {path} -- upstream may have changed; "
            "check manually")
    # write back; text mode restores the platform line ending on write
    open(path, "w", encoding="utf-8").write(s.replace(old, new, 1))
    print(f"  {path}: patched")


def cmd_patch(args):
    ensure_repo_root()

    # Patch A: Conformance C%
    f = "onpolicy/runner/shared/graph_mpe_runner.py"
    old_a = ('\t\tprint("Conformance_percentage Mean:", conformance_percentage_mean)\n'
             '\t\tprint("Conformance_percentage median:", conformance_percentage_median)\n')
    new_a = ('\t\t# raw = violation fraction (steps an agent left the corridor after entering).\n'
             '\t\t# Paper C% = 1 - violation, higher is better.\n'
             '\t\tprint("Conformance_percentage Mean (violation frac, raw):", conformance_percentage_mean)\n'
             '\t\tprint("Conformance_percentage median (violation frac, raw):", conformance_percentage_median)\n'
             '\t\tprint("Conformance C%% Mean (higher=better):  %.2f%%" % ((1.0 - conformance_percentage_mean) * 100.0))\n'
             '\t\tprint("Conformance C%% median (higher=better): %.2f%%" % ((1.0 - conformance_percentage_median) * 100.0))\n')
    _patch_file(f, old_a, new_a, marker="higher=better")

    # Patch B: vsync=False
    f = "multiagent/rendering.py"
    old_b = "self.window = pyglet.window.Window(width=width, height=height, display=display)"
    new_b = "self.window = pyglet.window.Window(width=width, height=height, display=display, vsync=False)"
    _patch_file(f, old_b, new_b, marker="vsync=False")

    print("\nOK: patches done. Next:  python repro.py check")


# ---------- subcommand: check ----------
def cmd_check(args):
    ensure_repo_root()
    py = need_venv()

    print(">>> [1/3] GPU / torch")
    verify_gpu(py)

    print(">>> [2/3] Pretrained weights")
    for f in ("model_weights/actor.pt", "model_weights/critic.pt", "model_weights/config.yaml"):
        if Path(f).exists():
            print("  ok ", f)
        else:
            die(f"missing {f}")

    print(">>> [3/3] End-to-end minimal eval (Single, 2 episodes)")
    cmd = eval_cmd(py, "sanity", "working_three_phase_graph", "5", "5", "120", "2")
    res = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    hits = [ln for ln in res.stdout.splitlines()
            if any(k in ln for k in ("Success rates mean", "Conformance C%", "Num collisions"))]
    if not hits:
        print(res.stdout[-2000:])
        print(res.stderr[-2000:], file=sys.stderr)
        die("no metric lines captured -- eval likely errored (see output above)")
    for ln in hits:
        print("  ", ln)
    print("\nOK: sanity check passed. Next:  python repro.py eval")


# ---------- subcommand: eval ----------
def cmd_eval(args):
    ensure_repo_root()
    py = need_venv()
    episodes = str(args.episodes)
    seed = str(args.seed)
    ts = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    out_path = Path(f"repro_results_{ts}.txt")

    header = (f"# Reproduction results  {datetime.datetime.now()}\n"
              f"# EPISODES={episodes}  weights=model_weights/ (obs16, single-corridor N5 training)\n")
    print(header)
    with open(out_path, "w", encoding="utf-8") as fout:
        fout.write(header + "\n")
        for tag, scenario, n, world, eplen in SCENARIOS:
            title = f"===== {tag}  (scenario={scenario} N={n} ep={eplen} episodes={episodes}) ====="
            print(title, flush=True)
            fout.write(title + "\n")
            cmd = eval_cmd(py, tag, scenario, str(n), str(world), str(eplen), episodes, seed=seed)
            res = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
            hits = [ln for ln in res.stdout.splitlines()
                    if any(k in ln for k in METRIC_KEYS)]
            if not hits:
                msg = "  (!! no metrics captured for this run; re-run it standalone to debug)"
                print(msg)
                fout.write(msg + "\n")
                # keep the tail of raw output for debugging
                fout.write(res.stdout[-1500:] + "\n" + res.stderr[-800:] + "\n")
            else:
                for ln in hits:
                    print("  ", ln)
                    fout.write(ln + "\n")
            print(flush=True)
            fout.write("\n")

    print(f"OK: all done. Summary saved to: {out_path}")
    print("   Compare against the reference values in REPRODUCTION_GUIDE.md section 6.")


# ---------- CBF eval command builder (added; used by `reproduce`) ----------
def cbf_eval_cmd(py, stage, filt, scenario, n, eplen, episodes, seed, args):
    """One eval_mpe.py command for a single density + seed, with the CBF metrics
    collector attached (and the safety filter attached only when filt != none).
    Drives the SAME render() loop as `eval`, with the SAME canonical protocol.
    `args` is the per-run shim built in cmd_reproduce."""
    cmd = [
        py, "-u", "onpolicy/scripts/eval_mpe.py",
        "--model_dir", stage["weights"],
        "--env_name", "GraphMPE", "--algorithm_name", "rmappo",
        "--scenario_name", CBF_SCENARIOS[scenario], "--dynamics_type", "air_taxi",
        "--num_agents", str(n), "--num_landmarks", str(n),
        "--world_size", str(args.world_size), "--episode_length", str(eplen),
        "--render_episodes", str(episodes), "--seed", str(seed),
        # canonical repro.py eval protocol (same flags `eval` passes)
        "--formation_type", "point", "--total_actions", "9", "--zeroshift", "10",
        "--use_dones", "False", "--ifi", "0.0", "--model_name", stage["tag"],
        # CBF: always record the rich metrics; attach the filter only when filtering
        "--cbf_eval",
        "--cbf_barrier", args.barrier, "--cbf_alpha", str(args.alpha),
        "--cbf_horizon", str(args.horizon), "--cbf_fallback", args.fallback,
        "--run_stage", stage["tag"], "--policy_slug", stage["policy"],
        "--eval_tag", stage["tag"],
    ]
    if filt != "none":
        cmd += ["--cbf_filter",
                "--cbf_neighbour_model", CBF_FILTER_NEIGHBOUR[filt]]
        # Rung-1 geometry-adaptive alpha (alpha-only ramp; horizon fixed).
        if getattr(args, "adaptive", False):
            cmd += ["--cbf_adaptive",
                    "--cbf_alpha_max", str(getattr(args, "alpha_max", 2.0)),
                    "--cbf_ell_thresh", str(getattr(args, "ell_thresh", 0.32)),
                    "--cbf_ell_hi", str(getattr(args, "ell_hi", 0.80)),
                    "--cbf_heading_gate", str(getattr(args, "heading_gate", 0.6))]
            _sr = getattr(args, "sense_range", None)
            if _sr is not None:
                cmd += ["--cbf_sense_range", str(_sr)]
        if getattr(args, "adaptive_horizon", False):
            cmd += ["--cbf_adaptive_horizon",
                    "--cbf_h_min", str(getattr(args, "h_min", 1))]
        if getattr(args, "gap_guard", False):
            cmd += ["--cbf_gap_guard"]   # Variant 1 junction gap guard
    if getattr(args, "show", False):
        cmd += ["--cbf_show"]
    return cmd


# ---------- subcommand: reproduce (added) ----------
# One command that re-runs the whole design-logic chain EVAL-SIDE on the EXISTING
# weights, so the order of thinking (Stage 2 -> horizon behaviour -> 3A failure ->
# 3B fix) is re-established on the robust/consistent harness. No training: the
# policy/barrier checkpoints are the design-history artifacts and are reused, not
# regenerated. Every run goes through the same cbf_eval_cmd -> render() path, so
# the baseline rows are identical by construction. Default 1 seed
# (fast logical check); pass --num_seeds 3 for the final error-bar data.
#
# Phases (run independently with --phase, or all at once):
#   stage2   original policy x filter{none,static,cv} x N{5,8,10}   (horizon 3)
#   horizon  original policy x neighbour{static,cv} x horizon{1,2,3} x N{5,8,10}
#   stage3a  3A policy x filter{none,cv} x N=5        (filter-dependence diagnostic)
#   stage3b  3B policy x filter{none,static,cv} x N{5,8,10}
# The 2x2 deployable factorial (policy{original,3B} x filter{static,cv}) is
# assembled from the stage2 (original) and stage3b (3B) result files.

# policy checkpoints — the design-history artifacts; reused, NEVER retrained
REPRO_POLICIES = {
    "original": dict(weights="model_weights",             policy="original"),
    "s3A":      dict(weights=CBF_STAGES["s3A"]["weights"], policy="shielded-train"),
    "s3B":      dict(weights=CBF_STAGES["s3B"]["weights"], policy="shaped-train"),
}
REPRO_HORIZONS = [1, 2, 3]
REPRO_NS = [5, 8, 10]


def _repro_matrix(phase, horizons=None, filters=None, alphas=None, ns=None):
    """List of run-specs dict(tag, policy_key, filt, n, horizon, fallback, alpha).
    The tag is the leading token of the result filename, so each phase's files are
    self-labelling. Phases (run standalone, or `all` for the full sequential set):

      baseline   original / no filter          x N            (also used, with
                 --scenario single, as the single-corridor anchor)
      stage2     original x {static,cv}         x N  (h=3)     filter efficacy
      horizon    original x static h{1,2} + cv h{1,2,4,5} x N  interaction + CV optimum
                 (h=3 <- stage2; CV swept past 3 to find the collision minimum)
      horizon3b  3B       x static h{1,2,3} + cv h{1,2,3,4,5} x N  deployed-policy optimum
      stage3     3A x {none,cv} (N=5) ; 3B/none x N            joint-training diagnostics
                 (3B filtered sits in horizon3b at h=3)
      fallback   3B x cv, cede-control fallback x N  (h=3)     no-safe-action ablation
      alpha      3B x cv, alpha in {0.5,0.75}  x N=10 (h=3)    class-K gain sweep

    Opt-in phases (never part of `all`; run with --scenario of choice):
      transfer   3B x cv at the deployed h on another scenario (generalisation)
      mergefix   3B x cv, horizon x alpha grid                (regime-split evidence)
      adaptive   3B x cv, Rung-1 adaptive alpha, fixed h      (tag AD_h3)
      adaptive_h 3B x cv, Rung-2 adaptive alpha + horizon     (tag ADH_h3; add
                 --gap_guard for the FINAL algorithm, tag ADH_h3_gap)
      headline   3B x cv, fixed refs h3a1 / h1a2 / h1a1.5 + Rung-2 adaptive

    `horizons` overrides the swept horizons for a standalone horizon/horizon3b run
    (applied to BOTH filters); in `all` each filter uses its own default set."""
    runs = []
    NS = ns or REPRO_NS

    def add(tag, pol, filt, n, h=3, fb="least_unsafe", al=1.0, adaptive=False,
            adaptive_horizon=False):
        runs.append(dict(tag=tag, policy_key=pol, filt=filt, n=n, horizon=h,
                         fallback=fb, alpha=al, adaptive=adaptive,
                         adaptive_horizon=adaptive_horizon))

    def inc(*names):
        return phase == "all" or phase in names

    # restrict the neighbour models for the FILTERED phases (stage2/horizon/
    # horizon3b); default is both. e.g. --filters cv runs CV-only.
    def filt_set(default):
        return tuple(f for f in default if filters is None or f in filters)

    if inc("baseline"):
        for n in NS:
            add("BASE", "original", "none", n)
    if inc("stage2"):
        for filt in filt_set(("static", "cv")):
            for n in NS:
                add("S2", "original", filt, n, h=3)
    # Per-filter horizon sweeps. CV is extended past h=3 to LOCATE the collision
    # optimum: CV collisions are still falling through h=3 (3.99->1.92->1.11 at
    # N=10), so 3 is the lowest sampled, not a proven minimum. Static is swept only
    # low, where the interaction figure needs it -- static rises monotonically with
    # horizon (compounding prediction error), so there is no optimum to find there.
    # h=3 for the ORIGINAL policy is supplied by stage2 (S2), so it is omitted here.
    HZ_ORIG = {"static": [1, 2], "cv": [1, 2, 4, 5]}
    HZ_3B = {"static": [1, 2, 3], "cv": [1, 2, 3, 4, 5]}
    if inc("horizon"):
        over = horizons if (horizons and phase == "horizon") else None
        for filt in filt_set(("static", "cv")):
            for hz in (over or HZ_ORIG[filt]):
                for n in NS:
                    add(f"HZ_h{hz}", "original", filt, n, h=hz)
    if inc("horizon3b"):
        over = horizons if (horizons and phase == "horizon3b") else None
        for filt in filt_set(("static", "cv")):
            for hz in (over or HZ_3B[filt]):
                for n in NS:
                    add(f"H3B_h{hz}", "s3B", filt, n, h=hz)
    if inc("stage3", "stage3a"):
        for filt in ("none", "cv"):
            add("S3A", "s3A", filt, 5, h=3)
    if inc("stage3", "stage3b"):
        for n in NS:
            add("S3B", "s3B", "none", n)
    if inc("fallback"):
        for n in NS:
            add("FB", "s3B", "cv", n, h=3, fb="unfiltered")
    if inc("alpha"):
        for al in (0.5, 0.75):
            add(f"A{al}", "s3B", "cv", 10, h=3, al=al)
    if phase == "transfer":  # opt-in only; deliberately NOT part of `all`
        # Deployed config (3B + CV) run on a NON-sequential scenario, to show the
        # sequential-developed filter still reduces collisions elsewhere
        # (generalisation / no regression). NOT part of `all`: run it standalone
        # with --scenario single|merge AFTER the sequential sweep names the optimum
        # horizon, passed via --horizons (defaults to 3, the current deployed h).
        ht = horizons[0] if horizons else 3
        for n in NS:
            add(f"TR_h{ht}", "s3B", "cv", n, h=ht)
    if phase == "mergefix":  # opt-in only; NOT part of `all`
        # Deployed policy (3B + CV) swept over a horizon x alpha grid to see how
        # much of the merge completion collapse (a CBF liveness/deadlock failure)
        # a pure parameter change buys back: SHORTER horizon = less pre-emptive
        # masking, HIGHER alpha = more permissive barrier. Run with
        # --scenario merge. Grid set by --horizons / --alphas (and --Ns to triage
        # at one density first).
        hs = horizons or [1, 2, 3]
        als = alphas or [1.0, 1.5, 2.0]
        for hz in hs:
            for al in als:
                for n in NS:
                    add(f"MF_h{hz}_a{al}", "s3B", "cv", n, h=hz, al=al)
    if phase == "adaptive":  # opt-in only; NOT part of `all`
        # Rung 1 of the geometry-adaptive filter: deployed policy (3B + CV),
        # fixed deployed horizon h=3, with alpha RAMPED per agent from the base
        # --alpha toward --alpha_max on the local conflict-geometry signal.
        # Unit-test it on the SEPARATE scenarios first (--scenario sequential,
        # then --scenario merge). Tag records the ramp window. --horizons picks
        # the fixed horizon (default 3), --Ns the density (default REPRO_NS).
        hz = (horizons or [3])[0]
        for n in NS:
            add(f"AD_h{hz}", "s3B", "cv", n, h=hz, al=1.0, adaptive=True)
    if phase == "adaptive_h":  # opt-in only; NOT part of `all`
        # Rung 2: proactive alpha (Rung 1) PLUS proactive horizon adaptation on the
        # same conflict-geometry signal (horizon ramps deployed->h_min with
        # convergence). Tag ADH so it never collides with the Rung-1 AD files.
        hz = (horizons or [3])[0]
        for n in NS:
            add(f"ADH_h{hz}", "s3B", "cv", n, h=hz, al=1.0, adaptive=True,
                adaptive_horizon=True)
    if phase == "headline":  # opt-in only; NOT part of `all`
        # One-shot headline comparison for the COMBINED scenario (run with
        # --scenario combined): the adaptive filter vs the three fixed baselines
        # it must beat or match. All on the deployed 3B policy + CV neighbour
        # model, same policy/scenario, so only the filter differs. Distinct tags
        # so the four never collide in one results folder.
        for n in NS:
            add("HLfix_h3a1",   "s3B", "cv", n, h=3, al=1.0)                  # deployed (seq-optimal)
            add("HLfix_h1a2",   "s3B", "cv", n, h=1, al=2.0)                  # merge-optimal
            add("HLcomp_h1a15", "s3B", "cv", n, h=1, al=1.5)                  # single compromise
            add("HLadapt",      "s3B", "cv", n, h=3, al=1.0,
                adaptive=True, adaptive_horizon=True)                          # Rung-2 adaptive
    return runs


def cmd_reproduce(args):
    import csv, time
    from collections import deque
    ensure_repo_root()
    py = need_venv()
    scenario = args.scenario
    eplen = args.episode_length or CBF_EPLEN[scenario]
    if args.seeds:
        seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]
    else:
        seeds = [args.seed + i for i in range(max(1, args.num_seeds))]
    horizons = [int(x) for x in str(args.horizons).split(",") if x.strip() != ""] or None
    filters = [f.strip() for f in str(args.filters).split(",") if f.strip() != ""] or None
    alphas = [float(x) for x in str(args.alphas).split(",") if x.strip() != ""] or None
    nsel = [int(x) for x in str(args.Ns).split(",") if x.strip() != ""] or None
    runs = _repro_matrix(args.phase, horizons, filters, alphas, nsel)
    full = [(spec, s) for spec in runs for s in seeds]

    print("=" * 70)
    print(f" REPRODUCE  phase={args.phase}  scenario={scenario} (episode_length={eplen})")
    print(f"            seeds={seeds}   episodes={args.episodes}")
    print(f"            configs={len(runs)}   total runs={len(full)}")
    print("=" * 70)

    # Build every run up-front (resolved command + label + manifest row), so the
    # manifest exists before anything executes and the job pool has the full list.
    jobs_list, manifest = [], []
    for i, (spec, s) in enumerate(full, 1):
        pol = REPRO_POLICIES[spec["policy_key"]]
        _tag = spec["tag"]
        if getattr(args, "gap_guard", False):
            # distinguish the gap-guarded run from the un-guarded baseline.
            _tag = f"{_tag}_gap"
        st = dict(tag=_tag, policy=pol["policy"], weights=pol["weights"])
        shim = argparse.Namespace(
            world_size=args.world_size, barrier=args.barrier,
            alpha=spec.get("alpha", args.alpha), horizon=spec["horizon"],
            fallback=spec.get("fallback", args.fallback),
            show=getattr(args, "show", False), device="cpu",
            adaptive=spec.get("adaptive", False),
            alpha_max=getattr(args, "alpha_max", 2.0),
            ell_thresh=getattr(args, "ell_thresh", 0.32),
            ell_hi=getattr(args, "ell_hi", 0.80),
            sense_range=getattr(args, "sense_range", None),
            heading_gate=getattr(args, "heading_gate", 0.6),
            adaptive_horizon=spec.get("adaptive_horizon", False),
            h_min=getattr(args, "h_min", 1),
            gap_guard=getattr(args, "gap_guard", False))
        cmd = [str(c) for c in cbf_eval_cmd(py, st, spec["filt"], scenario,
               spec["n"], eplen, args.episodes, s, shim)]
        label = (f"{spec['tag']} {spec['policy_key']}/{spec['filt']} "
                 f"N={spec['n']} h={spec['horizon']} seed={s}")
        logname = (f"{spec['tag']}_{spec['policy_key']}_{spec['filt']}_"
                   f"N{spec['n']}_h{spec['horizon']}_seed{s}.log")
        jobs_list.append(dict(idx=i, cmd=cmd, label=label, log=logname))
        manifest.append(dict(label=label, **spec, seed=s))

    man_path = (f"repro_manifest_{args.phase}_"
                f"{datetime.datetime.now():%y%m%d_%H%M%S}.csv")
    with open(man_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["label", "tag", "policy_key", "filt",
                                          "n", "horizon", "fallback", "alpha", "seed"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(manifest)
    print(f"manifest -> {man_path}")

    if args.dry_run:
        for j in jobs_list:
            print(f"\n----- run {j['idx']}/{len(jobs_list)}:  {j['label']} -----")
            print("    (dry run) " + " ".join(j["cmd"]))
        print("\nDRY RUN complete (nothing executed).")
        return

    jobs = max(1, int(args.jobs))
    total, rc = len(jobs_list), 0

    if jobs == 1:
        # Serial: stream each run's output straight to the console (watch it live).
        for j in jobs_list:
            print(f"\n----- run {j['idx']}/{total}:  {j['label']} -----", flush=True)
            r = subprocess.run(j["cmd"])
            if r.returncode != 0:
                print(f"    FAILED (exit {r.returncode}); continuing")
                rc = r.returncode
    else:
        # Parallel: up to `jobs` subprocesses at once, each writing to its own log
        # file. Result files are uniquely named per config, so they never collide;
        # the only shared resource is CPU/RAM, which is the point (fill idle cores).
        logdir = "repro_logs"
        os.makedirs(logdir, exist_ok=True)
        print(f"running {total} configs, up to {jobs} at a time; "
              f"per-run output -> {logdir}/")
        pending, running, done = deque(jobs_list), [], 0
        while pending or running:
            while pending and len(running) < jobs:
                j = pending.popleft()
                lf = open(os.path.join(logdir, j["log"]), "w", encoding="utf-8")
                p = subprocess.Popen(j["cmd"], stdout=lf, stderr=subprocess.STDOUT)
                running.append((p, j, lf))
                print(f"[start {j['idx']:>2}/{total}] {j['label']}", flush=True)
            for entry in list(running):
                p, j, lf = entry
                if p.poll() is not None:
                    lf.close()
                    running.remove(entry)
                    done += 1
                    ok = "OK" if p.returncode == 0 else f"FAIL(exit {p.returncode})"
                    if p.returncode != 0:
                        rc = p.returncode
                    print(f"[done  {done:>2}/{total}] {j['label']}  -> {ok}  "
                          f"({logdir}/{j['log']})", flush=True)
            time.sleep(1.0)

    if rc == 0:
        print("\nOK: reproduce phase complete. One result file per run in the repo root.")
    else:
        print(f"\nDONE with failures (last exit {rc}); see logs / output above.")


# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser(
        description="Cross-platform reproduction driver (Windows + Linux).")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("setup", help="build the env with uv")
    s.add_argument("--stack", choices=["auto", "blackwell", "default"], default="auto",
                   help="which torch stack (auto = detect via nvidia-smi)")
    s.add_argument("--pyg-index",
                   default="https://data.pyg.org/whl/torch-2.0.1+cu117.html",
                   help="PyG wheel index for the DEFAULT stack (match your CUDA)")
    s.set_defaults(func=cmd_setup)

    sub.add_parser("patch", help="apply the two quality patches").set_defaults(func=cmd_patch)
    sub.add_parser("check", help="GPU / weights / minimal-eval self-check").set_defaults(func=cmd_check)

    e = sub.add_parser("eval", help="reproduce all evaluable scenarios")
    e.add_argument("--episodes", type=int, default=100,
                   help="episodes per scenario (default 100; use 30 for a quick smoke test)")
    e.add_argument("--seed", type=int, default=1,
                   help="random seed for evaluation")
    e.set_defaults(func=cmd_eval)

    # ---- reproduce (added): re-run the whole design chain eval-side ----
    r = sub.add_parser("reproduce",
        help="re-run the design-logic chain (Stage2 / horizon sweep / 3A / 3B) "
             "eval-side on the existing weights; no training")
    r.add_argument("--phase",
                   choices=["all", "baseline", "stage2", "horizon", "horizon3b",
                            "stage3", "stage3a", "stage3b", "fallback", "alpha",
                            "transfer", "mergefix", "adaptive", "adaptive_h",
                            "headline"],
                   default="all", help="which part of the chain to run")
    r.add_argument("--scenario", choices=list(CBF_SCENARIOS.keys()),
                   default="sequential", help="canonical chain scenario")
    r.add_argument("--episodes", type=int, default=50, help="episodes per run")
    r.add_argument("--seed", type=int, default=1, help="first seed")
    r.add_argument("--num_seeds", type=int, default=1,
                   help="seeds counting up from --seed (default 1 for the logical "
                        "check; use 3 for the final error-bar data)")
    r.add_argument("--seeds", type=str, default=None,
                   help="explicit comma list, e.g. 1,2,3 (overrides --num_seeds)")
    r.add_argument("--filters", type=str, default="",
                   help="restrict neighbour models in the filtered phases "
                        "(stage2/horizon/horizon3b), e.g. 'cv' for CV-only or "
                        "'static,cv' (default: both)")
    r.add_argument("--alpha", type=float, default=1.0, help="CBF class-K gain")
    r.add_argument("--alphas", type=str, default="",
                   help="alpha grid for the mergefix phase, e.g. '1.0,1.5,2.0' "
                        "(default: 1.0,1.5,2.0)")
    r.add_argument("--Ns", type=str, default="",
                   help="restrict densities for any phase, e.g. '10' to triage at "
                        "one density first (default: 5,8,10)")
    r.add_argument("--fallback", choices=["least_unsafe", "unfiltered"],
                   default="least_unsafe")
    r.add_argument("--barrier", type=str, default=CBF_BARRIER_DEFAULT)
    r.add_argument("--world_size", type=float, default=5.0)
    r.add_argument("--episode_length", type=int, default=None,
                   help="override; default per-scenario (single=120, else=200)")
    r.add_argument("--horizons", type=str, default="",
                   help="override swept horizons for a standalone horizon/horizon3b "
                        "run (comma list). Left empty, horizon uses 1,2 (h=3 from "
                        "stage2) and horizon3b uses 1,2,3.")
    # --- Rung-1 geometry-adaptive alpha (phase 'adaptive') ---------------------
    r.add_argument("--alpha_max", type=float, default=2.0,
                   help="upper end of the adaptive alpha ramp (phase 'adaptive'); "
                        "ramps from --alpha up to this. Default 2.0.")
    r.add_argument("--ell_thresh", type=float, default=0.32,
                   help="conflict angle (rad) at/below which the adaptive ramp is "
                        "full (Chandra et al. liveness threshold ~0.32).")
    r.add_argument("--ell_hi", type=float, default=0.80,
                   help="conflict angle (rad) above which the adaptive ramp is off.")
    r.add_argument("--sense_range", type=float, default=None,
                   help="km; only neighbours within this range feed the adaptive "
                        "signal (default: all agents).")
    r.add_argument("--heading_gate", type=float, default=0.6,
                   help="rad; neighbours whose heading is within this of the ego's "
                        "are treated as in-line following and excluded from the "
                        "adaptive ramp (corridor queueing stays at base alpha; only "
                        "crossing/opposing geometry ramps). Default 0.6 (~34 deg). "
                        "Pass 0 for the ungated Rung-1 behaviour.")
    r.add_argument("--h_min", type=int, default=1,
                   help="shortest horizon the Rung-2 (phase 'adaptive_h') horizon "
                        "ramp can reach. Default 1.")
    r.add_argument("--gap_guard", action="store_true",
                   help="Variant 1: enable the junction gap guard on an adaptive "
                        "filter (e.g. --phase adaptive_h) -- suppress the ramp when "
                        "an agent is outside its corridor tube. The tag gets a _gap "
                        "suffix so guarded files don't collide with the un-guarded run.")
    r.add_argument("--jobs", type=int, default=1,
                   help="run this many evals concurrently (default 1 = serial, "
                        "streamed to the console). Use 3-4 on a 6-core box to cut "
                        "wall-clock ~2-3x; each parallel run logs to repro_logs/. "
                        "Lower it if RAM is tight (~1.5-2.5 GB per run).")
    r.add_argument("--dry_run", action="store_true",
                   help="print the full run matrix + commands, execute nothing")
    r.add_argument("--show", action="store_true",
                   help="draw the live per-step render window while evaluating "
                        "(passes --cbf_show). For visually sanity-checking a "
                        "scenario; use with a single config and few episodes, "
                        "e.g. --phase adaptive_h --episodes 2 --show. Metrics are "
                        "identical either way; this only controls the display.")
    r.set_defaults(func=cmd_reproduce)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
