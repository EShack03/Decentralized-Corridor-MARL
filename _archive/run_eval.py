#!/usr/bin/env python3
# ============================================================
# run_eval.py - one-line CBF evaluation driver (Windows + Linux)
# ------------------------------------------------------------
# The slick front door to the CBF safety evaluation. Mirrors repro.py: you give
# it the two things that change run-to-run (agent count, episodes) and it fills
# in everything else from a named --stage, forces the canonical repro.py eval
# protocol, runs over your seeds, and writes a result file whose NAME says what
# produced it.
#
# Typical use (everything else defaulted - final system, N=10, 50 ep, 3 seeds):
#   python run_eval.py --num_agents 10 --episodes 50
#
# Go back a stage, or drop the filter, by name:
#   python run_eval.py --num_agents 5 --episodes 50 --stage original   # pure baseline
#   python run_eval.py --num_agents 10 --stage s2                      # shield on original policy
#   python run_eval.py --num_agents 10 --stage s3B --filter none       # 3B policy, no shield
#
# Seeds - any count you like:
#   python run_eval.py --num_agents 10 --num_seeds 2                    # seeds 1,2
#   python run_eval.py --num_agents 10 --seeds 1,2,3,4                  # explicit list
#
# Run from the ROOT of the repo (where onpolicy/ lives), venv built via
# `python repro.py setup`.
# ============================================================
import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_MARKER = Path("onpolicy/scripts/eval_cbf_stage2.py")

# ------------------------------------------------------------------
# STAGE TABLE - the single source of truth for "what each stage is".
# Each stage maps to: the policy weights folder, a human policy slug (used in
# the result filename), the workflow tag, and the filter that stage uses by
# default. Edit the `weights` paths here if your run folders differ; nothing
# else in the driver hard-codes them.
#
#   stage          what it is                                 policy weights
#   -----          ----------                                 --------------
#   original       baseline InforMARL policy, no safety layer model_weights/
#   s2             original policy + eval-time shield          model_weights/
#   s3A            shielded-in-training policy (the failure)   stage3a run folder
#   s3B            reward-shaped policy (the fix)              stage3b run folder
#
# s2 is "original weights + a filter": same policy as `original`, so it shares
# the model_weights/ folder and only differs by the filter being on.
# ------------------------------------------------------------------
STAGES = {
    "original": dict(
        tag="s0", policy_slug="original",
        weights="model_weights",
        default_filter="none",
    ),
    "s2": dict(
        tag="s2", policy_slug="original",
        weights="model_weights",
        default_filter="cv",
    ),
    "s3A": dict(
        tag="s3A", policy_slug="shielded-train",
        weights=os.path.join("onpolicy", "results", "GraphMPE",
                             "three_phase_graph_updated", "rmappo",
                             "stage3a_full", "run2", "models"),
        default_filter="cv",
    ),
    "s3B": dict(
        tag="s3B", policy_slug="shaped-train",
        weights=os.path.join("onpolicy", "results", "GraphMPE",
                             "working_three_phase_graph", "rmappo",
                             "stage3b_full_v2", "run1", "models"),
        default_filter="cv",
    ),
}

# scenario short name -> the env's scenario_name
SCENARIOS = {
    "single": "working_three_phase_graph",
    "sequential": "three_phase_graph_sequential",
    "merge": "three_phase_graph_merge",
}

# filter slug -> the harness's neighbour-model value ("none" means no filter)
FILTER_TO_NEIGHBOUR = {"static": "static", "cv": "constant_velocity"}

# episode_length by scenario (matches repro.py: single=120, multi-corridor=200)
EPLEN = {"single": 120, "sequential": 200, "merge": 200}

BARRIER_DEFAULT = "cbf_stage1_barrier.pt"


def _show(cmd):
    """Render a command list for display only, so each argument is unambiguous.
    Every token is wrapped in double quotes so flags and their values never read
    as glued together (e.g. '--seed 1 --cbf_barrier', not '--seed 1--cbf_barrier').
    Purely cosmetic -- subprocess.run is given the list itself, never this string,
    so this cannot affect execution."""
    return " ".join(f'"{tok}"' for tok in cmd)


def die(msg):
    print("ERROR:", msg, file=sys.stderr)
    sys.exit(1)


def ensure_repo_root():
    if not REPO_MARKER.exists():
        die("run from the repo root (onpolicy/scripts/eval_cbf_stage2.py not found)")


def venv_python() -> str:
    py = (Path(".venv") / "Scripts" / "python.exe") if os.name == "nt" \
        else (Path(".venv") / "bin" / "python")
    if not py.exists():
        die(f"venv not found at {py} -- run  python repro.py setup  first")
    return str(py)


def resolve_seeds(args):
    """Explicit --seeds wins; else --num_seeds counts up from --seed; else one."""
    if args.seeds:
        return [int(x) for x in args.seeds.split(",") if x.strip() != ""]
    if args.num_seeds:
        n = max(1, int(args.num_seeds))
        return [int(args.seed) + i for i in range(n)]
    return [int(args.seed)]


def build_command(py, args):
    stage = STAGES[args.stage]

    # filter: explicit --filter overrides the stage default; `original` never filters
    filt = args.filter or stage["default_filter"]
    if args.stage == "original":
        filt = "none"

    scen_name = SCENARIOS[args.scenario]
    eplen = args.episode_length or EPLEN[args.scenario]
    seeds = resolve_seeds(args)

    cmd = [
        py, "-u", "onpolicy/scripts/eval_cbf_stage2.py",
        "--model_dir", stage["weights"],
        "--env_name", "GraphMPE", "--algorithm_name", "rmappo",
        "--scenario_name", scen_name, "--dynamics_type", "air_taxi",
        "--num_agents", str(args.num_agents),
        "--num_landmarks", str(args.num_agents),   # landmarks track agents (repro.py rule)
        "--world_size", str(args.world_size),
        "--episode_length", str(eplen),
        "--render_episodes", str(args.episodes),
        "--n_rollout_threads", str(args.n_rollout_threads),
        "--seeds", ",".join(str(s) for s in seeds),
        "--seed", str(seeds[0]),
        "--cbf_barrier", args.barrier,
        "--cbf_alpha", str(args.alpha),
        "--cbf_horizon", str(args.horizon),
        # naming passthroughs so the result filename reads as the convention
        "--run_stage", stage["tag"],
        "--policy_slug", stage["policy_slug"],
        "--eval_tag", stage["tag"],
    ]

    # The harness always evaluates baseline (no filter) vs filtered over the SAME
    # episodes and writes both columns. For --filter none we want a clean no-shield
    # answer: the BASELINE column already is exactly that (the raw policy), so we
    # force the filter's fallback to 'unfiltered', which makes the filtered column
    # collapse onto the baseline - both columns then agree and the file is an
    # unambiguous no-filter result. For static/cv we pass the real neighbour model.
    if filt == "none":
        cmd += ["--cbf_neighbour_model", "static", "--cbf_fallback", "unfiltered"]
    else:
        cmd += ["--cbf_neighbour_model", FILTER_TO_NEIGHBOUR[filt],
                "--cbf_fallback", args.fallback]

    # --with_baseline asks for an A/B of (filtered) vs (no filter). That only
    # means something when a filter is actually active: for a no-filter config
    # (original, or --filter none) both columns would be the same raw policy, so
    # we drop the flag and just report the single baseline result.
    if args.with_baseline and filt != "none":
        cmd += ["--with_baseline"]
    elif args.with_baseline and filt == "none":
        print("note: --with_baseline ignored for a no-filter run (nothing to "
              "compare against); reporting the single baseline result.")

    if args.density_sweep:
        cmd += ["--cbf_density_sweep", "--cbf_density_Ns", args.density_Ns]
    if args.design_sweep:
        cmd += ["--cbf_sweep"]

    return cmd, filt, seeds, eplen


def main():
    p = argparse.ArgumentParser(
        description="One-line CBF evaluation driver. You normally only set "
                    "--num_agents and --episodes; everything else has a default.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # --- the two you normally touch ---
    p.add_argument("--num_agents", type=int, default=10,
                   help="agent count / density")
    p.add_argument("--episodes", type=int, default=50,
                   help="episodes per configuration")

    # --- the headline knob: which stage of the system ---
    p.add_argument("--stage", choices=list(STAGES.keys()), default="s3B",
                   help="which system: original (baseline), s2 (shield on "
                        "original policy), s3A (shielded-in-training policy), "
                        "s3B (reward-shaped policy, the final system)")

    # --- optional overrides (sensible defaults per stage) ---
    p.add_argument("--filter", choices=["none", "static", "cv"], default=None,
                   help="eval-time shield: none / static / cv. Default follows "
                        "the stage (cv for s2/s3A/s3B, forced none for original)")
    p.add_argument("--scenario", choices=list(SCENARIOS.keys()), default="sequential",
                   help="corridor scenario")
    p.add_argument("--with_baseline", action="store_true",
                   help="also run the no-filter baseline and report the A/B "
                        "comparison. Default: run only the final filtered system")
    p.add_argument("--seeds", type=str, default=None,
                   help="explicit comma list, e.g. 1,2,3 (overrides --num_seeds)")
    p.add_argument("--num_seeds", type=int, default=1,
                   help="how many seeds, counting up from --seed (any count). "
                        "Default 1 (single run); pass --num_seeds 3 for error bars")
    p.add_argument("--seed", type=int, default=1,
                   help="first seed")

    # --- rarely touched; here for completeness ---
    p.add_argument("--alpha", type=float, default=1.0, help="CBF class-K gain")
    p.add_argument("--horizon", type=int, default=3, help="filter lookahead steps")
    p.add_argument("--fallback", choices=["least_unsafe", "unfiltered"],
                   default="least_unsafe", help="action when no primitive is safe")
    p.add_argument("--barrier", type=str, default=BARRIER_DEFAULT,
                   help="trained barrier checkpoint")
    p.add_argument("--world_size", type=float, default=5.0)
    p.add_argument("--episode_length", type=int, default=None,
                   help="override; default is per-scenario (single=120, else=200)")
    p.add_argument("--n_rollout_threads", type=int, default=8)

    # --- batch modes ---
    p.add_argument("--density_sweep", action="store_true",
                   help="run every N in --density_Ns in one go")
    p.add_argument("--density_Ns", type=str, default="5,8,10")
    p.add_argument("--design_sweep", action="store_true",
                   help="sweep alpha x horizon x fallback at the current N")

    p.add_argument("--dry_run", action="store_true",
                   help="print the resolved command and exit (run nothing)")

    args = p.parse_args()
    ensure_repo_root()
    py = venv_python()
    cmd, filt, seeds, eplen = build_command(py, args)

    st = STAGES[args.stage]
    print("=" * 70)
    print(f" stage     : {args.stage}  ({st['policy_slug']} policy, tag {st['tag']})")
    print(f" weights   : {st['weights']}")
    print(f" filter    : {filt}")
    print(f" scenario  : {args.scenario}  (N={args.num_agents}, episode_length={eplen})")
    print(f" episodes  : {args.episodes}   seeds: {seeds}")
    print(f" baseline  : {'yes (A/B comparison)' if args.with_baseline else 'no (filtered system only)'}")
    if args.density_sweep:
        print(f" MODE      : density sweep over N={args.density_Ns}")
    if args.design_sweep:
        print(" MODE      : design sweep (alpha x horizon x fallback)")
    print("=" * 70)

    if args.dry_run:
        print(">>> (dry run) would execute:\n    " + _show(cmd))
        return

    print(">>>", _show(cmd), flush=True)
    r = subprocess.run(cmd)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
