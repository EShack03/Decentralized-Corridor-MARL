#!/usr/bin/env python3
# ============================================================================
# results_index.py - one catalogue of every evaluation result in the repo
# ----------------------------------------------------------------------------
# Scans the repo root and run_results/ for evaluation result files
# (<tag>_<policy>_<filter>_<scenario>N<n>_<episodes>ep_seed<k>_<YYMMDD>.txt),
# reads the run settings and headline numbers out of each one, and writes
#
#   run_results/index.tsv     one row per result file (tab-separated, tracked
#                             by git, so it travels with the results)
#
# repro.py uses the same scan to SKIP any run whose result already exists, so a
# batch never repeats work by accident (pass --rerun to repro.py to force it).
#
#   python results_index.py                      # rebuild the index, print a summary
#   python results_index.py --find S2_ip seqN20  # list results whose name has ALL words
#   python results_index.py --check              # also report seeds rerun with different numbers
#
# Episode length: result files written after this patch carry a "# config:" line
# with episode_length and the filter settings. Older files do not; their episode
# length is shown as blank, and repro.py treats a match with a blank length as
# "already have it" (it says so, and --rerun overrides).
# Diagnostic only: reads result files, never touches the simulation.
# ============================================================================
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

DEST = Path("run_results")
INDEX = DEST / "index.tsv"

NAME = re.compile(
    r"^(?P<tag>.+?)_(?P<policy>original|shaped-train|shielded-train)_(?P<filt>none|static|cv)_"
    r"(?P<scen>.+?)N(?P<n>\d+)_(?P<episodes>\d+)ep_(?P<seed>seed\d+|\d+seed)_(?P<date>\d{6})\.txt$")

# short scenario names used in the file names -> the names repro.py uses
SCEN = {
    "seq": "sequential", "merge": "merge", "single": "single", "split": "split",
    "three_phase_graph_merge_sequential": "mergeseq",
    "three_phase_graph_merge_sequential_hard": "mergeseqhard",
    "three_phase_graph_split_and_merge": "combined",
    "three_phase_graph_combined_splits_and_merges_and_sequential": "combinedfull",
}

# (label at the start of a result line, column name)
METRICS = [
    ("collisions (LoS)", "los"),
    ("collisions cross-check", "los_xchk"),
    ("mean success rate", "success_pct"),
    ("corridor conformance", "conformance_pct"),
    ("collision-free episodes", "collision_free_pct"),
    ("min separation (km)", "min_sep_km"),
    ("integrated intrusion (km.st)", "integrated_intrusion"),
    ("throughput (agents/min)", "throughput"),
    ("completion time (s)", "completion_s"),
    ("collisions in corridor", "los_in"),
    ("collisions outside corridor", "los_out"),
]
COLUMNS = (["file", "folder", "status", "tag", "policy", "filter", "scenario", "n",
            "episodes", "seed", "date", "episode_length", "horizon", "alpha", "fallback"]
           + [c for _, c in METRICS] + ["no_safe_pct", "commit"])
_NUM = re.compile(r"-?\d+\.?\d*")


def status_of(path):
    """'superseded' / 'smoke' files never count as 'already have it'."""
    parts = set(path.parts)
    if "_superseded" in parts:
        return "superseded"
    if "09_smoke_tests" in parts:
        return "smoke"
    return "current"


def parse_result(path):
    """-> dict of settings + headline numbers, or None if not a result file."""
    m = NAME.match(path.name)
    if not m:
        return None
    row = dict(file=path.name, folder=str(path.parent).replace("\\", "/"),
               status=status_of(path), tag=m["tag"], policy=m["policy"],
               filter=m["filt"], scenario=SCEN.get(m["scen"], m["scen"]),
               n=int(m["n"]), episodes=int(m["episodes"]), seed=m["seed"],
               date=m["date"], episode_length="", horizon="", alpha="", fallback="",
               no_safe_pct="", commit="")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return row
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("# config:"):
            for k, v in re.findall(r"(\w+)=(\S+)", s):
                if k == "episode_length":
                    row["episode_length"] = v
                elif k == "commit":
                    row["commit"] = v
            continue
        mf = re.search(r"filter:.*?alpha=([\d.]+).*?horizon=(\d+).*?fallback=(\w+)", s)
        if mf:
            row["alpha"], row["horizon"], row["fallback"] = mf.group(1), mf.group(2), mf.group(3)
            continue
        if s.startswith("no-safe-action rate"):
            mn = re.search(r"([\d.]+)%", s)
            row["no_safe_pct"] = mn.group(1) if mn else ""
            continue
        for label, col in METRICS:
            if s.startswith(label):
                mn = _NUM.search(s[len(label):])
                if mn:
                    row[col] = mn.group(0)
                break
    if row["horizon"] == "" and row["filter"] == "none":
        row["horizon"] = "-"
    return row


def scan(root=Path(".")):
    """Every result file in the repo root and anywhere under run_results/."""
    paths = [p for p in root.iterdir() if p.is_file()]
    if (root / DEST).is_dir():
        paths += [p for p in (root / DEST).rglob("*.txt")]
    rows = []
    for p in sorted(set(paths)):
        r = parse_result(p)
        if r:
            rows.append(r)
    return rows


def run_key(tag, policy, filt, scenario, n, episodes, seed):
    """What makes two results 'the same run' (episode length is checked separately)."""
    return (tag, policy, filt, SCEN.get(scenario, scenario), int(n), int(episodes), str(seed))


def existing_runs(root=Path(".")):
    """run_key -> list of current (not superseded, not smoke) result rows."""
    have = defaultdict(list)
    for r in scan(root):
        if r["status"] == "current":
            have[run_key(r["tag"], r["policy"], r["filter"], r["scenario"],
                         r["n"], r["episodes"], r["seed"])].append(r)
    return have


def find_existing(have, key, episode_length):
    """The first existing result for this run at this episode length (a result
    with no recorded episode length counts as a match), else None."""
    for r in have.get(key, []):
        if r["episode_length"] in ("", str(episode_length)):
            return r
    return None


def write_index(rows, path=INDEX):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(COLUMNS) + "\n")
        for r in sorted(rows, key=lambda r: (r["status"] != "current", r["tag"], r["scenario"],
                                             r["n"], r["seed"], r["date"])):
            f.write("\t".join(str(r.get(c, "")) for c in COLUMNS) + "\n")


def main():
    ap = argparse.ArgumentParser(description="catalogue every evaluation result")
    ap.add_argument("--find", nargs="+", default=None,
                    help="only list results whose file name contains ALL of these words")
    ap.add_argument("--check", action="store_true",
                    help="report runs that exist more than once with different numbers")
    a = ap.parse_args()
    if not Path("onpolicy/scripts/eval_mpe.py").exists():
        sys.exit("run from the repo root")
    rows = scan()
    if a.find:
        hits = [r for r in rows if all(w in r["file"] for w in a.find)]
        for r in hits:
            print(f"{r['folder']}/{r['file']}   eplen={r['episode_length'] or '?'}  "
                  f"LoS={r.get('los_xchk', '?')}  success={r.get('success_pct', '?')}%  [{r['status']}]")
        print(f"{len(hits)} match(es)")
        return
    write_index(rows)
    cur = [r for r in rows if r["status"] == "current"]
    print(f"indexed {len(rows)} result files ({len(cur)} current) -> {INDEX}")
    groups = defaultdict(list)
    for r in cur:
        groups[run_key(r["tag"], r["policy"], r["filter"], r["scenario"], r["n"],
                       r["episodes"], r["seed"]) + (r["episode_length"],)].append(r)
    dups = {k: v for k, v in groups.items() if len(v) > 1}
    if dups:
        print(f"{len(dups)} run(s) exist more than once (same settings and seed):")
        for k, v in sorted(dups.items()):
            same = len({(r.get("los_xchk"), r.get("success_pct")) for r in v}) == 1
            if a.check or not same:
                print(f"  {'identical' if same else 'DIFFERENT'}: " + ", ".join(r["file"] for r in v))
        if not a.check:
            print("  (identical repeats confirm determinism; --check lists them all)")


if __name__ == "__main__":
    main()
