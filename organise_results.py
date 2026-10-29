#!/usr/bin/env python3
# ============================================================================
# organise_results.py - file every evaluation result into run_results/
# ----------------------------------------------------------------------------
# New result files are written to the repo root by every eval run. This script
# moves them (and the older result folders) into one tidy tree, grouped by
# study, mirroring the sections of the Run Results Ledger:
#
#   run_results/
#     01_stage_pipeline/            Stage 2 -> 3A -> 3B chain, 3 seeds (Ledger 1)
#     02_horizon_sweep_3B/          3B policy horizon sweep, 100 ep (Ledger 1.3)
#     03_adaptive_rung1/            Rung-1 adaptive alpha (AD_*)
#     04_fixed_vs_adaptive/<scen>/  fixed settings + Rung-2 adaptive (Ledger 2.1-2.3, 2.6)
#     05_stall_trigger/             Rung-3 stall trigger, superseded (Ledger 2.4)
#     06_gap_guard/<scen>/          FINAL algorithm: adaptive + gap guard (Ledger 2.5, 2.7)
#     07_no_filter/<scen>/          policies with no filter: BASE (original), S3B (3B)
#     08_other/<scen>/              anything else with a recognised result name
#     09_smoke_tests/               runs of 5 episodes or fewer
#     _superseded/                  config-bug-era, early smoke and superseded files
#     manifests/                    repro_manifest_*.csv run lists (local only)
#
# Each result .txt keeps its per-episode twin (<name>_perep.csv) beside it.
# Tracked files are moved with `git mv` (history is kept); untracked files
# (e.g. the per-episode .csv files, which .gitignore excludes) are moved normally.
#
# Safe by default: without --apply it only PRINTS the plan.
#   python organise_results.py            # show what would move
#   python organise_results.py --apply    # do it
# Re-run after every batch of evals: it only touches files not yet filed.
# ============================================================================
import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

DEST = Path("run_results")

NAME = re.compile(
    r"^(?P<tag>.+?)_(?P<pol>original|shaped-train|shielded-train)_(?P<filt>none|static|cv)_"
    r"(?P<scen>.+?)N(?P<n>\d+)_(?P<ep>\d+)ep_(?P<seed>seed\d+|\d+seed)_(?P<date>\d{6})"
    r"(?P<perep>_perep)?\.(?P<ext>txt|csv)$")

SCEN = {
    "seq": "sequential", "merge": "merge", "single": "single", "split": "split",
    "three_phase_graph_merge_sequential": "mergeseq",
    "three_phase_graph_merge_sequential_hard": "mergeseqhard",
    "three_phase_graph_split_and_merge": "combined",
    "three_phase_graph_combined_splits_and_merges_and_sequential": "combinedfull",
}

# whole legacy folders -> new home (moved file by file, keeping sub-folders)
LEGACY = {
    "results_archive_pre100": DEST / "01_stage_pipeline",
    "results_ungated_AD": DEST / "03_adaptive_rung1",
    "_archive/config_bug_era": DEST / "_superseded" / "config_bug_era",
    "_archive/smoke_runs": DEST / "_superseded" / "smoke_runs",
    "_archive/superseded_results": DEST / "_superseded" / "superseded_results",
}
# single legacy files at the root or in _archive/
LEGACY_FILES = {
    "sweep_t03.txt": DEST / "05_stall_trigger",
    "sweep_t05.txt": DEST / "05_stall_trigger",
    "sweep_t08.txt": DEST / "05_stall_trigger",
    "repro_report.md": DEST,
}


README = """# run_results/ — every evaluation result, filed by study

New runs write their result files to the repo root; `python organise_results.py --apply`
files them here (re-run it after each batch). Folder numbers follow the Run Results Ledger.

| Folder | What is in it | Ledger |
|---|---|---|
| `01_stage_pipeline/` | Stage 2 -> 3A -> 3B chain on the sequential corridor: original policy with no / static / constant-velocity filter (S2), horizon sweep of the original policy (HZ), 3A and 3B policies (S3A, S3B), 3 seeds x 50 episodes; plus the early density sweeps and reproduction checks | 1.0-1.7 |
| `02_horizon_sweep_3B/` | 3B policy, horizon 1-5, static vs constant velocity, 100 episodes (H3B) | 1.3 |
| `03_adaptive_rung1/` | Rung-1 adaptive alpha only (AD) | - |
| `04_fixed_vs_adaptive/<scenario>/` | fixed settings h3/a1, h1/a2, h1/a1.5 (HLfix, HLcomp) and the Rung-2 adaptive filter without the gap guard (HLadapt, ADH_h3) | 2.1-2.3, 2.6 |
| `05_stall_trigger/` | Rung-3 stall trigger and its threshold sweep (superseded) | 2.4 |
| `06_gap_guard/<scenario>/` | **final algorithm**: Rung-2 adaptive + junction gap guard (ADH_h3_gap) | 2.5, 2.7 |
| `07_no_filter/<scenario>/` | no filter: original policy (BASE) and 3B policy (S3B) | 2.5-2.7 |
| `08_other/` | any other recognised result | - |
| `09_smoke_tests/` | runs of 5 episodes or fewer | - |
| `_superseded/` | config-bug era, early smoke runs, superseded sweeps | provenance doc |
| `manifests/` | run lists written by `repro.py reproduce` (local only, not in git) | - |

**File names:** `<config>_<policy>_<filter>_<scenario>N<agents>_<episodes>ep_seed<k>_<YYMMDD>.txt`,
e.g. `ADH_h3_gap_shaped-train_cv_mergeN15_30ep_seed1_261025.txt` = final algorithm, 3B policy,
constant-velocity filter, merge, 15 aircraft, 30 episodes, seed 1, run on 25 Oct 2026.
Each `.txt` has a per-episode twin `<same name>_perep.csv` (kept locally; `.gitignore` excludes csv).
Scenarios: `seq` = sequential, `merge`, `single`, `three_phase_graph_merge_sequential` = mergeseq.
Note: the episode length is not in the name; see the Ledger for which runs used 200 s or 300 s.

**Collating:** `collate_repro.py` reads one folder, e.g.
`python collate_repro.py --dir run_results/06_gap_guard/mergeseq`.
"""


def classify(m):
    """Destination folder for a parsed result filename."""
    tag, scen = m["tag"], SCEN.get(m["scen"], m["scen"])
    if int(m["ep"]) <= 5:
        return DEST / "09_smoke_tests"
    if tag.startswith("H3B_"):
        return DEST / "02_horizon_sweep_3B"
    if tag.startswith("AD_"):
        return DEST / "03_adaptive_rung1"
    if tag.startswith("HLadapt_stall"):
        return DEST / "05_stall_trigger"
    if tag.endswith("_gap"):
        return DEST / "06_gap_guard" / scen
    if tag.startswith(("HLfix", "HLcomp", "HLadapt", "ADH_")):
        return DEST / "04_fixed_vs_adaptive" / scen
    if m["filt"] == "none":
        return DEST / "07_no_filter" / scen
    if tag.startswith(("S2", "HZ", "S3A", "S3B", "s0", "s2", "s3", "FB", "A0", "TR_", "MF_")):
        return DEST / "01_stage_pipeline" / ("extra_" + scen if scen != "sequential" else "")
    return DEST / "08_other" / scen


def tracked_files():
    try:
        out = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout
        return {Path(p) for p in out.splitlines()}
    except Exception:
        return set()


def plan_moves():
    moves = []
    # 1) result files at the repo root
    for p in sorted(Path(".").iterdir()):
        if not p.is_file():
            continue
        m = NAME.match(p.name)
        if m:
            moves.append((p, classify(m) / p.name))
        elif p.name.startswith("repro_manifest_") and p.suffix == ".csv":
            moves.append((p, DEST / "manifests" / p.name))
        elif p.name in LEGACY_FILES:
            moves.append((p, LEGACY_FILES[p.name] / p.name))
    # 2) loose result files directly in _archive/
    arch = Path("_archive")
    if arch.is_dir():
        for p in sorted(arch.iterdir()):
            if p.is_file() and p.suffix in (".txt", ".csv"):
                moves.append((p, DEST / "_superseded" / "archive_root" / p.name))
    # 3) legacy result folders
    for src, dst in LEGACY.items():
        s = Path(src)
        if s.is_dir():
            for p in sorted(s.rglob("*")):
                if p.is_file():
                    moves.append((p, dst / p.relative_to(s)))
    return moves


def main():
    ap = argparse.ArgumentParser(description="file evaluation results into run_results/")
    ap.add_argument("--apply", action="store_true", help="actually move the files")
    args = ap.parse_args()
    if not Path("onpolicy/scripts/eval_mpe.py").exists():
        sys.exit("run from the repo root")

    moves = [(s, d) for s, d in plan_moves() if s.resolve() != d.resolve()]
    if not moves:
        print("Nothing to organise: every result is already filed.")
        return
    tracked = tracked_files()
    by_dir = {}
    for s, d in moves:
        by_dir.setdefault(str(d.parent), []).append(s.name)
    for d in sorted(by_dir):
        print(f"{d}/  <- {len(by_dir[d])} file(s)")
        for n in by_dir[d][:4]:
            print(f"      {n}")
        if len(by_dir[d]) > 4:
            print(f"      ... and {len(by_dir[d]) - 4} more")
    print(f"\n{len(moves)} file(s) to move "
          f"({sum(1 for s, _ in moves if s in tracked)} tracked by git -> git mv).")
    if not args.apply:
        print("Dry run only. Re-run with --apply to move them.")
        return

    for s, d in moves:
        d.parent.mkdir(parents=True, exist_ok=True)
        if d.exists():
            print(f"  SKIP (already exists): {d}")
            continue
        if s in tracked:
            subprocess.run(["git", "mv", str(s), str(d)], check=True)
        else:
            shutil.move(str(s), str(d))
    # remove legacy folders left empty
    for src in list(LEGACY) + ["_archive"]:
        for root, dirs, files in os.walk(src, topdown=False):
            if not os.listdir(root):
                os.rmdir(root)
    readme = DEST / "README.md"
    if not readme.exists():
        readme.write_text(README, encoding="utf-8")
        print(f"wrote {readme}")
    print("Done. Review with `git status`, then commit.")


if __name__ == "__main__":
    main()
