#!/usr/bin/env python3
# ============================================================================
# analyse_traces.py - where do the unsuccessful aircraft end up?
# ----------------------------------------------------------------------------
# Reads the per-step traces written with CBF_TRACE=1 (cbf_traces/*.csv, one file
# per run and episode) and classifies every aircraft that did NOT succeed by
# where it is and what stage it reached when the episode ended:
#
#   Q    still before corridor 1, never inside it (queue / spawn)
#   S1   entered corridor 1 through its SIDE (never a valid entry, never counts)
#   C1   validly inside corridor 1 when time ran out (congestion)
#   X1   validly entered corridor 1 but left it without a valid exit (no hand-off)
#   G    handed off to corridor 2 but still in the junction gap, never entered it
#   S2   entered corridor 2 through its SIDE (cut the corner; never counts)
#   C2   validly inside corridor 2 when time ran out
#   X2   validly entered corridor 2 but left it without a valid exit
#
# "Valid entry" mirrors the scenario's gate: the first in-corridor sample must be
# within the first 8 % of the corridor (+ the 0.05 km tolerance). Success mirrors
# the runner: last corridor and phase_reached >= 2.
#
#   python analyse_traces.py                    # every run in cbf_traces/
#   python analyse_traces.py --match seqN20     # only file names containing this
# Writes nothing; prints one table per run (tag + scenario + N + seed).
# ============================================================================
import argparse
import csv
import glob
import os
import re
from collections import Counter, defaultdict

GATE_FRONT = 0.08   # scenario gate_front_ratio
EPS = 0.05          # scenario gate tolerance (km)
CATS = ["Q", "S1", "C1", "X1", "G", "S2", "C2", "X2", "other"]
DESC = {"Q": "never reached corridor 1 (queue/spawn)",
        "S1": "entered corridor 1 through its side",
        "C1": "inside corridor 1 at the end",
        "X1": "left corridor 1 without a valid exit",
        "G": "in the junction gap, never entered corridor 2",
        "S2": "entered corridor 2 through its side",
        "C2": "inside corridor 2 at the end",
        "X2": "left corridor 2 without a valid exit",
        "other": "unclassified"}


def f(x):
    try:
        return float(x)
    except ValueError:
        return float("nan")


def classify(rows):
    """rows: one aircraft's trace rows in step order -> (success, category, info)."""
    last = rows[-1]
    tb, ph, nt = int(last["current_tube"]), int(last["phase_reached"]), int(last["n_tubes"])
    if tb < 0 or nt <= 0:
        return None, "other", {}
    success = (tb == nt - 1 and ph >= 2) or int(last["status"]) == 1
    if success:
        return True, None, {}
    first_in = {}
    for r in rows:                       # first in-corridor sample per corridor
        k = int(r["current_tube"])
        if int(r["in_tube"]) == 1 and k not in first_in:
            first_in[k] = f(r["s_along"]), f(r["L"])
    def side(k):
        s, L = first_in[k]
        return s > GATE_FRONT * L + EPS
    s_end, L_end, in_end = f(last["s_along"]), f(last["L"]), int(last["in_tube"])
    if tb == 0:
        if ph == 0:
            cat = "S1" if (0 in first_in and side(0)) else ("Q" if 0 not in first_in else "other")
        else:
            cat = "C1" if in_end == 1 else "X1"
    else:
        if ph == 0:
            cat = "S2" if (tb in first_in and side(tb)) else ("G" if tb not in first_in else "other")
        else:
            cat = "C2" if in_end == 1 else "X2"
    return False, cat, {"s_end": s_end, "L": L_end, "lat_end": f(last["lat"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="cbf_traces")
    ap.add_argument("--match", default="")
    a = ap.parse_args()
    files = sorted(p for p in glob.glob(os.path.join(a.dir, "*.csv")) if a.match in os.path.basename(p))
    runs = defaultdict(list)
    for p in files:
        runs[re.sub(r"_ep\d+\.csv$", "", os.path.basename(p))].append(p)
    if not runs:
        print("no trace files found"); return
    for run, paths in sorted(runs.items()):
        cats, n_air, n_succ = Counter(), 0, 0
        for p in paths:
            per = defaultdict(list)
            with open(p, newline="") as fh:
                for r in csv.DictReader(fh):
                    if "phase_reached" not in r:
                        break                      # old-format trace, skip
                    per[int(r["agent"])].append(r)
            for ag, rows in per.items():
                rows.sort(key=lambda r: int(r["step"]))
                ok, cat, _ = classify(rows)
                if ok is None:
                    continue
                n_air += 1
                if ok:
                    n_succ += 1
                else:
                    cats[cat] += 1
        if n_air == 0:
            print(f"\n{run}: no usable rows"); continue
        n_fail = n_air - n_succ
        print(f"\n{run}  ({len(paths)} episodes, {n_air} aircraft)")
        print(f"  success {n_succ}/{n_air} = {100.0 * n_succ / n_air:.1f}%   failures {n_fail}")
        for c in CATS:
            if cats[c]:
                print(f"  {c:<5} {cats[c]:>5}  {100.0 * cats[c] / max(n_fail, 1):5.1f}% of failures   {DESC[c]}")


if __name__ == "__main__":
    main()
