#!/usr/bin/env python3
# ============================================================================
# trace_where.py - WHERE along the corridor do unsuccessful aircraft leave it?
# ----------------------------------------------------------------------------
# Companion to analyse_traces.py (same cbf_traces/*.csv, same failure classes).
# For every unsuccessful aircraft it finds the moment that decided the failure:
#
#   X1 / X2  left a corridor without a valid exit: the LAST step it was inside
#            that corridor, i.e. where it left. Reports how far along the
#            corridor (s/L) and which way it left:
#              side-early  out of a side wall before 90 % of the length
#              side-late   out of a side wall in the last 10 % (cutting towards
#                          the goal / the next corridor before the exit gate)
#              end-wide    past the end but outside the exit gate's width
#              back        out of the entrance end (turned back)
#   S1 / S2  entered a corridor through its side: the FIRST step inside it,
#            i.e. how far along it cut in (s/L).
#
#   python trace_where.py                    # every run in cbf_traces/
#   python trace_where.py --match seqN20     # only file names containing this
# Writes nothing; prints one table per run. Diagnostic only.
# ============================================================================
import argparse
import csv
import glob
import os
import re
from collections import Counter, defaultdict

from analyse_traces import classify, f

BINS = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.9), (0.9, 10.0)]
BIN_NAMES = ["0-25%", "25-50%", "50-75%", "75-90%", "90-100%"]


def bin_of(frac):
    if frac < 0:
        return None
    for (lo, hi), name in zip(BINS, BIN_NAMES):
        if lo <= frac < hi:
            return name
    return BIN_NAMES[-1]


def where(rows, cat):
    """-> (s/L, how) for the deciding moment of this failure, or None."""
    tb = int(rows[-1]["current_tube"])
    inside = [r for r in rows if int(r["current_tube"]) == tb and int(r["in_tube"]) == 1]
    if not inside:
        return None
    if cat in ("S1", "S2"):
        r = inside[0]
        L = f(r["L"])
        return f(r["s_along"]) / L if L > 0 else None, "cut in"
    # X1 / X2: last step inside, then the first step after it says which way it went
    last = inside[-1]
    k = rows.index(last)
    nxt = rows[k + 1] if k + 1 < len(rows) else last
    L, hw = f(last["L"]), f(last["half_width"])
    s_last = f(last["s_along"])
    s_n, lat_n = f(nxt["s_along"]), abs(f(nxt["lat"]))
    if s_n < 0:
        how = "back"
    elif s_n > L:
        how = "end-wide"
    elif lat_n > hw:
        how = "side-late" if (L > 0 and s_last / L >= 0.9) else "side-early"
    else:
        how = "other"
    return (s_last / L if L > 0 else None), how


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
        pos = defaultdict(Counter)      # cat -> bin -> count
        how = defaultdict(Counter)      # cat -> how -> count
        for p in paths:
            per = defaultdict(list)
            with open(p, newline="") as fh:
                for r in csv.DictReader(fh):
                    if "phase_reached" not in r:
                        break
                    per[int(r["agent"])].append(r)
            for ag, rows in per.items():
                rows.sort(key=lambda r: int(r["step"]))
                ok, cat, _ = classify(rows)
                if ok is not False or cat not in ("X1", "X2", "S1", "S2"):
                    continue
                w = where(rows, cat)
                if w is None or w[0] is None:
                    continue
                b = bin_of(w[0])
                if b:
                    pos[cat][b] += 1
                how[cat][w[1]] += 1
        if not pos:
            continue
        print(f"\n{run}  ({len(paths)} episodes)")
        print(f"  {'class':<6}{'n':>5}   " + "".join(f"{n:>9}" for n in BIN_NAMES) + "   how")
        for cat in ("S1", "X1", "S2", "X2"):
            n = sum(pos[cat].values())
            if not n:
                continue
            hows = ", ".join(f"{k} {v}" for k, v in how[cat].most_common())
            print(f"  {cat:<6}{n:>5}   " + "".join(f"{pos[cat][b]:>9}" for b in BIN_NAMES) + f"   {hows}")
        print("  (columns: how far along the corridor it left (X) or cut in (S), as a share of its length)")


if __name__ == "__main__":
    main()
