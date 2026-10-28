#!/usr/bin/env python3
# ============================================================================
# collate_repro.py - collate the `repro.py reproduce` result files into one
# old-vs-new comparison report, plus the horizon-sweep panel.
# ----------------------------------------------------------------------------
# Standalone, standard-library only (no torch / numpy needed), so it runs in
# the venv or any Python 3. Point it at the folder holding the result files
# (default: current dir). It:
#   * parses every  <TAG>_<policy>_<filt>_<scen>N<N>_<eps>ep_<seeds>_<date>.txt
#   * averages across seeds (so a 1-seed or 3-seed run both work)
#   * writes  repro_parsed.csv   (every metric, every config - for your own plots)
#   * writes  repro_report.md    (old-vs-new tables + reconciliation + horizon panel)
#   * prints a short reconciliation summary to the console
#
# The OLD reference numbers are the documented pre-repro 3-seed means (Stage 2,
# 3A, 3B result docs). The horizon panel is NEW data only - it is the result
# the sweep exists to establish, so there is no old table to put beside it.
#
#   python collate_repro.py                 # parse ./  -> repro_report.md
#   python collate_repro.py --dir results   # parse a specific folder
# ============================================================================
import argparse
import csv
import glob
import os
import re
import statistics
import sys

# ---- documented pre-repro reference means (policy, filter, N) -> metrics ----
# policy label: "original" | "3A" | "3B".  filter: "none" | "static" | "cv".
OLD = {
    ("original", "none", 5):  dict(coll=11.72, succ=89.5, conf=99.4),
    ("original", "none", 8):  dict(coll=27.57, succ=90.2),
    ("original", "none", 10): dict(coll=32.85, succ=87.6),
    ("original", "static", 5):  dict(coll=2.47,  succ=82.9),
    ("original", "static", 8):  dict(coll=12.61, succ=78.4),
    ("original", "static", 10): dict(coll=21.56, succ=76.1),
    ("original", "cv", 5):  dict(coll=0.05, succ=89.2, conf=98.5),
    ("original", "cv", 8):  dict(coll=0.50, succ=88.2, conf=97.0),
    ("original", "cv", 10): dict(coll=1.11, succ=86.7, conf=96.1),
    ("3B", "none", 5):  dict(coll=6.13,  succ=95.3),
    ("3B", "none", 8):  dict(coll=22.54, succ=93.5),
    ("3B", "none", 10): dict(coll=43.39, succ=94.1),
    ("3B", "static", 5):  dict(coll=0.87,  succ=94.0),
    ("3B", "static", 8):  dict(coll=6.38,  succ=91.9),
    ("3B", "static", 10): dict(coll=14.33, succ=91.5),
    ("3B", "cv", 5):  dict(coll=0.05, succ=94.7),
    ("3B", "cv", 8):  dict(coll=0.55, succ=92.7),
    ("3B", "cv", 10): dict(coll=1.33, succ=92.7),
    ("3A", "none", 5): dict(coll=53.79, succ=68.8, conf=100.0),
    ("3A", "cv", 5):   dict(coll=0.06,  succ=64.7, conf=99.3),
}

# policy slug in the filename -> policy label used above
SLUG2POLICY = {"original": "original", "shaped-train": "3B", "shielded-train": "3A"}

# the metric labels in the result file -> short keys
METRICS = [
    ("collisions (LoS)", "coll"),
    ("collisions cross-check", "coll_xchk"),
    ("mean success rate", "succ"),
    ("corridor conformance", "conf"),
    ("collision-free episodes", "cfree"),
    ("min separation (km)", "min_sep"),
    ("mean closest approach (km)", "mean_closest"),
    ("max intrusion depth (km)", "max_intr"),
    ("integrated intrusion (km.st)", "int_intr"),
    ("near-miss rate %", "nearmiss"),
    ("spacing violations %", "spacing"),
    ("throughput (agents/min)", "throughput"),
    ("completion time (s)", "completion"),
    ("collisions in corridor", "coll_in"),
    ("collisions outside corridor", "coll_out"),
    ("max intrusion in (km)", "max_intr_in"),
    ("max intrusion out (km)", "max_intr_out"),
]
_NUM = re.compile(r"(-?\d+\.?\d*)")


def parse_filename(stem):
    """Right-anchored parse: ...the last six tokens are fixed, the tag is the
    rest (so a tag with an underscore, e.g. HZ_h1, is preserved)."""
    t = stem.split("_")
    if len(t) < 7:
        return None
    date, seedslug, epsep, scenN, filt, policy = t[-1], t[-2], t[-3], t[-4], t[-5], t[-6]
    tag = "_".join(t[:-6])
    mN = re.search(r"N(\d+)", scenN)
    meps = re.search(r"(\d+)ep", epsep)
    mscen = re.match(r"([a-z]+)N\d+", scenN)
    if not mN:
        return None
    return dict(tag=tag, policy_slug=policy, filt=filt,
                N=int(mN.group(1)),
                scen=mscen.group(1) if mscen else None,
                episodes=int(meps.group(1)) if meps else None,
                seedslug=seedslug, date=date, stem=stem)


def parse_file(path):
    meta = parse_filename(os.path.splitext(os.path.basename(path))[0])
    if meta is None:
        return None
    vals, horizon, neighbour, alpha, no_safe, blocked = {}, None, None, None, None, None
    fallback = None
    filtered = False
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.rstrip("\n")
            if "FILTERED SYSTEM" in s:
                filtered = True
            mflt = re.search(r"neighbour=(\w+).*?alpha=([\d.]+).*?horizon=(\d+)", s)
            if mflt:
                neighbour = mflt.group(1)
                alpha = float(mflt.group(2))
                horizon = int(mflt.group(3))
            mfb = re.search(r"fallback=(\w+)", s)
            if mfb:
                fallback = mfb.group(1)
            for label, key in METRICS:
                if s.strip().startswith(label):
                    m = _NUM.search(s[len(s) - len(s.lstrip()) + len(label):])
                    if m:
                        vals[key] = float(m.group(1))
                    break
            if s.startswith("no-safe-action rate"):
                m = re.search(r"([\d.]+)%", s)
                if m:
                    no_safe = float(m.group(1))
            if s.startswith("avg primitives blocked"):
                m = _NUM.search(s.split(":", 1)[-1])
                if m:
                    blocked = float(m.group(1))
    # derive horizon for non-filtered or HZ tag
    if horizon is None:
        mh = re.search(r"h(\d+)", meta["tag"])
        horizon = int(mh.group(1)) if mh else (None if not filtered else 3)
    row = dict(meta)
    row.update(vals)
    row["filtered"] = filtered
    row["horizon"] = horizon
    row["neighbour"] = neighbour
    row["alpha"] = alpha
    row["fallback"] = fallback
    row["no_safe_pct"] = no_safe
    row["avg_blocked"] = blocked
    row["policy"] = SLUG2POLICY.get(meta["policy_slug"], meta["policy_slug"])
    # seed count from slug (seed1 -> 1 seed; "3seed" -> 3)
    msd = re.match(r"(\d+)seed", meta["seedslug"])
    row["seed_from"] = meta["seedslug"]
    return row


def phase_of(tag):
    if tag.startswith("HZ"):
        return "horizon"
    return {"S2": "stage2", "S3A": "stage3a", "S3B": "stage3b"}.get(tag, tag)


def aggregate(rows):
    """Average metric keys across seeds, grouping by
    (scen,tag,policy,filt,N,horizon,alpha,fallback). Scenario and the
    alpha/fallback ablation axes MUST be in the key: without scen the single/merge
    baselines collapse into the seq ones; without alpha/fallback the A* and FB
    ablation runs collapse into the canonical deployed run."""
    groups = {}
    for r in rows:
        k = (r.get("scen"), r["tag"], r["policy"], r["filt"], r["N"],
             r["horizon"], r.get("alpha"), r.get("fallback"))
        groups.setdefault(k, []).append(r)
    agg = []
    numkeys = [k for _, k in METRICS] + ["no_safe_pct", "avg_blocked"]
    for k, rs in groups.items():
        scen, tag, policy, filt, N, horizon, alpha, fallback = k
        a = dict(scen=scen, tag=tag, phase=phase_of(tag), policy=policy, filt=filt,
                 N=N, horizon=horizon, alpha=alpha, fallback=fallback,
                 n_seeds=len(rs), neighbour=rs[0]["neighbour"],
                 filtered=rs[0]["filtered"])
        for nk in numkeys:
            xs = [r[nk] for r in rs if r.get(nk) is not None]
            a[nk] = round(statistics.mean(xs), 4) if xs else None
            a[nk + "_std"] = round(statistics.pstdev(xs), 4) if len(xs) > 1 else None
        agg.append(a)
    return agg


def _is_canonical(a):
    """True for a normal deployed run: default class-K gain and least-unsafe
    fallback. Excludes the alpha-sweep (A*) and fallback-ablation (FB) runs, which
    otherwise masquerade as the canonical run at the same (policy,filt,N,horizon)."""
    return (a.get("alpha") in (None, 1.0)
            and a.get("fallback") in (None, "least_unsafe"))


def find(agg, policy, filt, N, horizon=None, scen="seq"):
    """Return the canonical run for a config. Defaults to the sequential scenario
    (the design-chain scenario the OLD references were measured on) and to the
    deployed alpha/fallback; skips ablation runs so reconciliation and the stage
    tables cannot accidentally pick up an FB/alpha run."""
    for a in agg:
        if scen is not None and a.get("scen") not in (scen, None):
            continue
        if not _is_canonical(a):
            continue
        if a["policy"] == policy and a["filt"] == filt and a["N"] == N:
            if horizon is None or a["horizon"] == horizon:
                return a
    return None


def fmt(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def reconcile(agg):
    """Compare new vs OLD where a reference exists; flag out-of-band cells.
    Band: collisions within max(0.6 abs, 25% rel); success within 6 points."""
    lines, n_pass, n_check = [], 0, 0
    lines.append("| config | metric | old | new | Δ | verdict |")
    lines.append("|---|---|---|---|---|---|")
    for (policy, filt, N), ref in sorted(OLD.items()):
        a = find(agg, policy, filt, N, horizon=(3 if filt != "none" else None))
        if a is None:
            continue
        for key, lbl, band in (("coll", "collisions", "coll"),
                               ("succ", "success%", "succ")):
            if key not in ref or a.get(key) is None:
                continue
            old, new = ref[key], a[key]
            d = new - old
            if band == "coll":
                ok = abs(d) <= max(0.6, 0.25 * old)
            else:
                ok = abs(d) <= 6.0
            verdict = "PASS" if ok else "CHECK"
            n_pass += ok
            n_check += (not ok)
            lines.append(f"| {policy}/{filt} N={N} | {lbl} | {fmt(old)} | "
                         f"{fmt(new)} | {d:+.2f} | {verdict} |")
    return lines, n_pass, n_check


def table_oldnew(agg, title, policy, filts, Ns):
    out = [f"### {title}", "",
           "| N | " + " | ".join(f"{f} old→new coll" for f in filts) +
           " | " + " | ".join(f"{f} old→new succ%" for f in filts) + " |",
           "|---|" + "|".join(["---"] * (2 * len(filts))) + "|"]
    for N in Ns:
        cells = []
        for f in filts:
            ref = OLD.get((policy, f, N), {})
            a = find(agg, policy, f, N, horizon=(3 if f != "none" else None))
            co = fmt(ref.get("coll")); cn = fmt(a.get("coll") if a else None)
            cells.append(f"{co}→{cn}")
        for f in filts:
            ref = OLD.get((policy, f, N), {})
            a = find(agg, policy, f, N, horizon=(3 if f != "none" else None))
            so = fmt(ref.get("succ"), 1); sn = fmt(a.get("succ") if a else None, 1)
            cells.append(f"{so}→{sn}")
        out.append(f"| {N} | " + " | ".join(cells) + " |")
    out.append("")
    return out


def _horizon_table(agg, policy, policy_label, horizons=(1, 2, 3, 4, 5)):
    """One horizon table for a policy: collisions / no-safe% / success% by horizon
    and neighbour model. Canonical seq runs only. For the original policy, h=3 is
    the Stage-2 run (same config); for 3B every horizon is an H3B run."""
    pool = [a for a in agg if a.get("scen") in ("seq", None) and _is_canonical(a)
            and a["policy"] == policy and a["filt"] in ("static", "cv")
            and a["horizon"] in horizons]
    present = sorted({a["horizon"] for a in pool})
    if not present:
        return []
    out = [f"#### {policy_label} policy", "",
           "| neighbour | N | " + " | ".join(f"h={h}" for h in present) + " |",
           "|---|---|" + "|".join(["---"] * len(present)) + "|"]
    for neigh in ("static", "cv"):
        for N in (5, 8, 10):
            cells = []
            for h in present:
                a = next((x for x in pool if x["filt"] == neigh and x["N"] == N
                          and x["horizon"] == h), None)
                cells.append("-" if a is None else
                             f"{fmt(a.get('coll'))} / {fmt(a.get('no_safe_pct'),1)} "
                             f"/ {fmt(a.get('succ'),1)}")
            out.append(f"| {neigh} | {N} | " + " | ".join(cells) + " |")
    out.append("")
    return out


def horizon_panel(agg):
    swept = [a for a in agg if a.get("scen") in ("seq", None) and _is_canonical(a)
             and a["filt"] in ("static", "cv") and a["horizon"] in (1, 2, 4, 5)]
    if not swept:
        return ["### Horizon sweep", "",
                "_(no horizon-phase files yet; run `--phase horizon` / `--phase "
                "horizon3b`. h=3 is already present from the Stage-2 / H3B runs.)_", ""]
    out = ["### Horizon sweep (new data - the result the sweep establishes)", "",
           "collisions / no-safe-action% / success% by horizon and neighbour model. "
           "For the original policy h=3 is the Stage-2 run; longer CV horizons are the "
           "new HZ/H3B runs. Read along a row: static rises with horizon (compounding "
           "prediction error), CV falls then plateaus (useful look-ahead).", ""]
    out += _horizon_table(agg, "original", "Original")
    out += _horizon_table(agg, "3B", "3B (deployed)")
    return out


def determinism_panel(agg):
    """Same (policy, filter, N, horizon) present under more than one tag should be
    byte-identical (same seed, deterministic harness, read-only instrumentation).
    Any non-zero delta means hidden nondeterminism or the measurement perturbing
    the run - the exact 'observer must not disturb the system' check. The natural
    pair is S2 (h=3) vs HZ_h3 (regenerated h=3)."""
    from collections import defaultdict
    groups = defaultdict(list)
    for a in agg:
        # a true replica shares EVERY run parameter -- scenario, density, horizon
        # AND the alpha/fallback ablation axes -- differing only in the tag it was
        # filed under. Keying on all of them stops the FB/alpha ablation runs from
        # being mistaken for non-deterministic copies of the canonical run.
        groups[(a.get("scen"), a["policy"], a["filt"], a["N"], a["horizon"],
                a.get("alpha"), a.get("fallback"))].append(a)
    dups = {k: v for k, v in groups.items()
            if len({x["tag"] for x in v}) > 1}
    head = ["### Determinism / non-perturbation cross-check", ""]
    if not dups:
        return head + [
            "_(no duplicate configs in this matrix: each config is produced under a "
            "single tag, so there is nothing to cross-check here. Bit-exact "
            "determinism / non-perturbation was established separately in the 3-seed "
            "reproduction, where the regenerated h=3 runs matched the Stage-2 runs to "
            "every decimal.)_", ""]
    out = head + [
        "Identical config under two tags - should match to every decimal (same "
        "seed, deterministic harness, read-only metrics). Any Δ>0 flags "
        "nondeterminism or the instrumentation perturbing the run.", "",
        "| config | tags | collisions | success% | verdict |",
        "|---|---|---|---|---|"]
    for key, v in sorted(dups.items(), key=lambda kv: str(kv[0])):
        scen, policy, filt, N, h, alpha, fb = key
        tags = ",".join(sorted({x["tag"] for x in v}))
        colls = [x["coll"] for x in v if x.get("coll") is not None]
        succs = [x["succ"] for x in v if x.get("succ") is not None]
        dcoll = (max(colls) - min(colls)) if len(colls) > 1 else 0.0
        dsucc = (max(succs) - min(succs)) if len(succs) > 1 else 0.0
        verdict = ("IDENTICAL" if dcoll == 0 and dsucc == 0
                   else f"DIFFERS (Δcoll={dcoll:.4f}, Δsucc={dsucc:.3f})")
        out.append(f"| {policy}/{filt} N={N} h={h} | {tags} | "
                   f"{' / '.join(fmt(c, 3) for c in colls)} | "
                   f"{' / '.join(fmt(s, 2) for s in succs)} | {verdict} |")
    out.append("")
    return out


def pooled_panel(folder):
    """Pool every per-episode CSV for a config into ONE sample and report a
    proper standard error, the collision tail (percentiles / max), and the
    collision-free rate. This is what makes 'run N seeds x M episodes, treat as
    one N*M-episode run' exact for the spread, not just the mean."""
    files = sorted(glob.glob(os.path.join(folder, "*_perep.csv")))
    head = ["### Pooled per-episode statistics (all runs of a config concatenated)", ""]
    if not files:
        return head + [
            "_(no per-episode CSVs found. Re-run with the updated metrics file so "
            "each run emits <result>_perep.csv; then several seeds of one config "
            "pool into a single estimate here, with a real standard error and the "
            "safety tail.)_", ""]
    groups = {}
    for fp in files:
        stem = os.path.splitext(os.path.basename(fp))[0]
        if stem.endswith("_perep"):
            stem = stem[:-len("_perep")]
        meta = parse_filename(stem)
        if meta is None:
            continue
        tag = meta["tag"]
        mh = re.search(r"h(\d+)", tag)
        horizon = int(mh.group(1)) if mh else None
        # label the ablation runs so they are NOT pooled together (FB and the two
        # alpha runs all lack an h-token, so without this they would merge into one
        # meaningless row at the same scen/policy/filt/N).
        if tag.startswith("FB"):
            abl = "FB"
        elif re.match(r"A[0-9]", tag):
            abl = tag.split("_")[0]  # e.g. A0.5
        else:
            ma = re.search(r"_a([0-9.]+)", tag)  # mergefix MF_h1_a2.0
            abl = f"a{ma.group(1)}" if ma else ""
        key = (meta.get("scen"),
               SLUG2POLICY.get(meta["policy_slug"], meta["policy_slug"]),
               meta["filt"], meta["N"], horizon, abl)
        g = groups.setdefault(key, {"coll": [], "succ": [], "cfree": []})
        with open(fp, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    g["coll"].append(float(row["collisions"]))
                    g["succ"].append(float(row["success_pct"]))
                    g["cfree"].append(float(row["collision_free"]))
                except (KeyError, ValueError):
                    pass
    out = head + [
        "Each row concatenates every per-episode CSV for that config. SE = "
        "std/sqrt(n). p50/p95/max are the per-episode collision percentiles (the "
        "safety tail); coll-free = episodes with zero loss of separation.", "",
        "| config | n_ep | mean coll | SE | p50 | p95 | max | coll-free % | mean succ% |",
        "|---|---|---|---|---|---|---|---|---|"]

    def pct(sorted_xs, p):
        n = len(sorted_xs)
        if n == 1:
            return sorted_xs[0]
        return sorted_xs[min(n - 1, int(round(p / 100.0 * (n - 1))))]

    for key in sorted(groups, key=lambda k: str(k)):
        scen, policy, filt, N, horizon, abl = key
        c = sorted(groups[key]["coll"])
        n = len(c)
        if n == 0:
            continue
        mean = statistics.mean(c)
        se = (statistics.pstdev(c) / (n ** 0.5)) if n > 1 else 0.0
        cfree = 100.0 * statistics.mean(groups[key]["cfree"])
        succ = statistics.mean(groups[key]["succ"])
        hlbl = f" h={horizon}" if horizon else ""
        allbl = f" [{abl}]" if abl else ""
        slbl = f"{scen}:" if scen else ""
        out.append(f"| {slbl}{policy}/{filt} N={N}{hlbl}{allbl} | {n} | {mean:.3f} | {se:.3f} | "
                   f"{pct(c,50):.2f} | {pct(c,95):.2f} | {c[-1]:.2f} | "
                   f"{cfree:.1f} | {succ:.1f} |")
    out.append("")
    return out


# episode length per scenario short-name (must match CBF_EPLEN in repro.py)
SCEN_EPLEN = {"single": 120, "seq": 200, "sequential": 200, "merge": 200, "split": 200}


def scenario_panel(folder, show_rate=False, eplen=None, deployed_h=3):
    """Cross-scenario difficulty comparison. Headline number is RAW mean
    per-episode collisions, grouped by (scenario, policy, filter, N) so you can
    read the danger ordering of the scenarios at matched density. This is the
    evidence for 'sequential is the most unsafe scenario, hence the right
    hardening target'.

    Only the CANONICAL deployed run of each config is included: the alpha-sweep
    (A*) and fallback-ablation (FB) runs are excluded, and for a filtered config
    only the deployed horizon (`deployed_h`) is used -- otherwise a cell would
    blend several horizons and the ablations into one misleading mean.

    Raw counts are only comparable across scenarios when the episodes are the
    SAME length (collisions are summed over the episode). Run single/merge at
    --episode_length 200 to match the sequential length; then raw counts are
    directly comparable. If you ever mix lengths, pass show_rate=True to add a
    length-normalised LoS rate (per 1000 env steps) in parentheses; `eplen`
    overrides the per-scenario length used for that rate (pass the matched
    length, e.g. 200, if you overrode single)."""
    files = sorted(glob.glob(os.path.join(folder, "*_perep.csv")))
    head = ["### Cross-scenario difficulty (raw per-episode collisions)", ""]
    groups = {}
    scens = set()
    for fp in files:
        stem = os.path.splitext(os.path.basename(fp))[0]
        if stem.endswith("_perep"):
            stem = stem[:-len("_perep")]
        meta = parse_filename(stem)
        if meta is None or meta.get("scen") is None:
            continue
        tag = meta["tag"]
        # canonical-only: drop the alpha-sweep (A0.5/A0.75), fallback-ablation (FB)
        # and merge-fix grid (MF_*) runs, and for a filtered config keep only the
        # deployed horizon.
        if tag.startswith("FB") or tag.startswith("MF") or re.match(r"A[0-9]", tag):
            continue
        if meta["filt"] != "none":
            mh = re.search(r"h(\d+)", tag)
            if mh and int(mh.group(1)) != deployed_h:
                continue
        scens.add(meta["scen"])
        key = (SLUG2POLICY.get(meta["policy_slug"], meta["policy_slug"]),
               meta["filt"], meta["N"])
        g = groups.setdefault(key, {})  # scen -> list of per-ep collisions
        with open(fp, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    g.setdefault(meta["scen"], []).append(float(row["collisions"]))
                except (KeyError, ValueError):
                    pass
    if len(scens) < 2:
        return head + ["_(only one scenario present; run `--phase baseline` with "
                       "`--scenario single` and `--scenario merge` alongside the "
                       "sequential set to populate this panel.)_", ""]
    sc_order = [s for s in ("single", "merge", "seq", "sequential", "split") if s in scens]
    sc_order += [s for s in sorted(scens) if s not in sc_order]
    caption = ("Mean per-episode collisions (loss of separation). Compare DOWN a "
               "column (same policy/filter/N) across scenarios: higher = more unsafe. "
               "Raw counts are comparable only at equal episode length -- run "
               "single/merge at --episode_length 200 to match sequential.")
    if show_rate:
        caption += (" Parenthesised value is the length-normalised rate "
                    "(LoS per 1000 env steps).")
    out = head + [caption, "",
        "| policy/filter N | " + " | ".join(sc_order) + " |",
        "|---|" + "|".join(["---"] * len(sc_order)) + "|"]
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2])):
        policy, filt, N = key
        cells = []
        for sc in sc_order:
            xs = groups[key].get(sc)
            if not xs:
                cells.append("--")
                continue
            m = statistics.mean(xs)
            if show_rate:
                rate = 1000.0 * m / (eplen or SCEN_EPLEN.get(sc, 200))
                cells.append(f"{m:.2f} ({rate:.2f})")
            else:
                cells.append(f"{m:.2f}")
        out.append(f"| {policy}/{filt} N={N} | " + " | ".join(cells) + " |")
    out.append("")
    return out


def main():
    ap = argparse.ArgumentParser(description="collate repro.py reproduce results")
    ap.add_argument("--dir", default=".", help="folder with the result .txt files")
    ap.add_argument("--out", default="repro_report.md")
    ap.add_argument("--csv", default="repro_parsed.csv")
    ap.add_argument("--show-rate", action="store_true",
                    help="also show length-normalised LoS rate in the cross-scenario "
                         "panel (only needed if scenarios ran at different lengths)")
    ap.add_argument("--eplen", type=int, default=None,
                    help="episode length to use for the --show-rate divisor across all "
                         "scenarios (pass the matched length, e.g. 200)")
    ap.add_argument("--deployed-h", type=int, default=3,
                    help="deployed filter horizon; the cross-scenario panel uses only "
                         "the filtered runs at this horizon (default 3)")
    args = ap.parse_args()

    pats = ["BASE_*.txt", "S2_*.txt", "HZ_*.txt", "H3B_*.txt", "S3A_*.txt",
            "S3B_*.txt", "FB_*.txt", "A*_*.txt", "TR_*.txt"]
    files = sorted({p for pat in pats for p in glob.glob(os.path.join(args.dir, pat))})
    rows = [r for r in (parse_file(p) for p in files) if r]
    if not rows:
        print(f"no result files found in {args.dir!r} "
              f"(looked for S2_*/HZ_*/S3A_*/S3B_* .txt)")
        sys.exit(1)

    # raw CSV
    allkeys = ["stem", "phase", "policy", "policy_slug", "filt", "N", "horizon",
               "neighbour", "alpha", "episodes", "seed_from", "filtered"] + \
              [k for _, k in METRICS] + ["no_safe_pct", "avg_blocked"]
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=allkeys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            r = dict(r); r["phase"] = phase_of(r["tag"])
            w.writerow(r)

    agg = aggregate(rows)
    rec_lines, n_pass, n_check = reconcile(agg)

    md = [f"# Reproduction report", "",
          f"Parsed **{len(rows)}** result files -> **{len(agg)}** configs "
          f"(after seed-averaging) from `{os.path.abspath(args.dir)}`.", "",
          "Format `old→new` = documented pre-repro 3-seed mean → this run "
          "(seed-averaged). Collisions = losses of separation per episode.", "",
          "## Reconciliation (new vs documented)", ""]
    md += rec_lines
    md += ["", f"**{n_pass} PASS, {n_check} CHECK** "
           f"(band: collisions ±max(0.6, 25%); success ±6 pts).", "",
           "## Stage 2 - original policy", ""]
    md += table_oldnew(agg, "Original policy x filter", "original",
                       ["none", "static", "cv"], [5, 8, 10])
    md += ["## Stage 3A - filter-dependence diagnostic (N=5)", ""]
    md += table_oldnew(agg, "3A policy", "3A", ["none", "cv"], [5])
    md += ["## Stage 3B - shaped policy", ""]
    md += table_oldnew(agg, "3B policy x filter", "3B",
                       ["none", "static", "cv"], [5, 8, 10])
    md += ["## " + "Horizon", ""]
    md += horizon_panel(agg)
    md += ["## Determinism", ""]
    md += determinism_panel(agg)
    md += ["## Pooled statistics", ""]
    md += pooled_panel(args.dir)
    md += ["## Cross-scenario difficulty", ""]
    md += scenario_panel(args.dir, show_rate=args.show_rate, eplen=args.eplen,
                         deployed_h=args.deployed_h)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"parsed {len(rows)} files -> {len(agg)} configs")
    print(f"reconciliation: {n_pass} PASS, {n_check} CHECK")
    print(f"wrote {args.out} and {args.csv}")


if __name__ == "__main__":
    main()
