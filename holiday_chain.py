#!/usr/bin/env python3
"""
[ES] added (18 Nov 2026) - unattended constrained-training run (19-23 Nov, nobody at the server).

1. TRAIN   every arm x seed in parallel through train_lag.py (stages 10 -> 15 -> 20 aircraft). A run that
           fails is restarted once (train_lag.py skips finished stages); a run that fails twice is
           abandoned and the others carry on.
2. STATUS  every --status_hours: run_results/18_lagrangian/status.md (stage, last update's costs,
           budgets, multipliers, progress, failures) is committed and pushed.
3. EXPORT  each finished run -> model_weights_lag/<arm>/seed<k>/ (actor.pt, critic.pt, config.yaml copied
           from model_weights/, lag_state.pt, lag_log_N<n>.csv per stage; end-of-stage actors in stage_N<n>/).
4. EVAL    for every exported policy (evaluation seed = training seed): E1 alone (21 core + 6 new layouts
           at 30), E2 with the final filter (same 27), high density with the filter (7 cases, 20 episodes);
           plus the original policy as reference where results are missing (seed 4; new layouts alone;
           high density at 20 episodes). Longest runs first, --eval_workers at a time.
5. FILE    all new result files -> run_results/18_lagrangian/, commit and push; status.md says DONE.

Launch (repo root):
  ulimit -n 65536; OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup setsid .venv/bin/python -u holiday_chain.py --push \
      > holiday_chain.log 2>&1 &
Re-running the same command resumes: finished stages, exports and result files are skipped.
"""
import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PY = sys.executable
OUT = Path("run_results/18_lagrangian")
WOUT = Path("model_weights_lag")
SCEN0 = "working_three_phase_graph_hidegoals"          # train_lag.py files every stage under the first layout
SLUG = {"lag_final": "lagfinal", "lag_orig": "lagorig"}
ENV = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", GIT_TERMINAL_PROMPT="0")   # never wait for a password

# ---------------------------------------------------------------- evaluation cases (as E1/E2, 17-18 Nov)
SC = {"seq": "three_phase_graph_sequential_hidegoals", "single": "working_three_phase_graph_hidegoals",
      "merge": "three_phase_graph_merge", "mergeseq": "three_phase_graph_merge_sequential",
      "vertiport": "three_phase_graph_vertiport", "unbal": "three_phase_graph_merge_unbalanced",
      "river": "three_phase_graph_river2way", "triple": "three_phase_graph_triple_merge",
      "ladder": "three_phase_graph_ladder", "branch": "three_phase_graph_branch"}
CORE = [("seq", n) for n in (5, 10, 15, 20, 25, 30)] + [(l, n) for l in ("single", "merge", "mergeseq") for n in (10, 15, 20, 25, 30)]
NEW = [(l, 30) for l in ("vertiport", "unbal", "river", "triple", "ladder", "branch")]
DENSE = [("vertiport", 60), ("vertiport", 80), ("vertiport", 100), ("ladder", 80), ("ladder", 100), ("mergeseq", 100), ("single", 100)]
BASE = ("--model_dir {md} --env_name GraphMPE --algorithm_name rmappo --scenario_name {sc} --dynamics_type air_taxi "
        "--num_agents {n} --num_landmarks {n} --world_size 5.0 --episode_length {L} --render_episodes {E} --seed {s} "
        "--formation_type point --total_actions 9 --zeroshift 10 --use_dones False --ifi 0.0 --model_name {t} --cbf_eval "
        "--cbf_barrier cbf_stage1_barrier.pt --cbf_alpha 1.0 --cbf_horizon 3 --cbf_fallback least_unsafe --run_stage {t} "
        "--policy_slug {slug} --eval_tag {t} --use_ReLU")
FINAL = (" --cbf_filter --cbf_neighbour_model route_map --cbf_ignore_parked --cbf_no_barrier --cbf_sep_scale 1.2 "
         "--cbf_route_variant fix --cbf_exact_dynamics --cbf_sense_ground_vel --cbf_sense_pos_sigma 0 --cbf_sense_vel_sigma 0 "
         "--cbf_sense_delay 0 --cbf_sense_kalman --cbf_sense_compensate")
LENS = {"mergeseq": {10: 300, 15: 300, 20: 350, 25: 400, 30: 450}}


def eplen(lay, n):
    return {60: 850, 80: 1100, 100: 1350}.get(n) or LENS.get(lay, {}).get(n, 300 if n <= 20 else (400 if n == 25 else 450))


def log(msg):
    print(f"[chain {time.strftime('%d %b %H:%M')}] {msg}", flush=True)


def sh(cmd, logfile=None):
    if logfile:
        with open(logfile, "a") as f:
            return subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, env=ENV).returncode
    return subprocess.run(cmd, shell=True, env=ENV).returncode


# ---------------------------------------------------------------- helpers on train_lag.py's folders
def stage_dir(arm, n, seed):
    return Path("onpolicy/results/GraphMPE") / SCEN0 / "rmappo" / f"lag_{arm}_N{n}_seed{seed}"


def done_models(arm, n, seed):
    d = stage_dir(arm, n, seed)
    for run in sorted(d.glob("run*"), key=lambda p: int(p.name[3:] or 0), reverse=True):
        if (run / "models" / "DONE").exists():
            return run / "models"
    return None


def newest_run(arm, n, seed):
    runs = sorted(stage_dir(arm, n, seed).glob("run*"), key=lambda p: int(p.name[3:] or 0))
    return runs[-1] if runs else None


def last_row(run):
    if run is None:
        return None
    f = run / "lag_log.csv"
    if not f.exists():
        return None
    try:
        rows = list(csv.DictReader(open(f)))
        return (len(rows), rows[-1]) if rows else None
    except Exception:
        return None


# ---------------------------------------------------------------- git
def git_push(paths, message, a):
    if not a.push:
        log(f"(no --push) would commit {paths}: {message}")
        return
    for p in paths:
        sh(f"git add -f {p}", "holiday_git.log")
    if sh(f'git commit -q -m "{message}"', "holiday_git.log") != 0:
        log("nothing new to commit")
    for k in range(3):
        if sh("git push -q", "holiday_git.log") == 0:
            log(f"pushed: {message}")
            return
        time.sleep(60)
    log("push failed three times (see holiday_git.log); will retry at the next status")


# ---------------------------------------------------------------- status
def write_status(a, state, phase, extra=""):
    OUT.mkdir(parents=True, exist_ok=True)
    upd_per_stage = int(a.steps) // (a.threads * a.episode_length)
    L = [f"# Constrained training - status ({time.strftime('%a %d %b %Y %H:%M')} server time)", "",
         f"Phase: **{phase}**. Started {state['start']}. Stages {a.stages}, {a.steps:.0f} steps per stage "
         f"(~{upd_per_stage} updates), {a.threads} workers, episodes {a.episode_length} s, PPO epochs {a.ppo_epoch}.", "",
         "| Arm | Seed | Stage | Updates in stage | Warm-up | J sep (budget, lambda) | J corr (budget, lambda) | Fallback | KL | State |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for arm in a.arms:
        for s in a.seeds:
            st = state["runs"].get(f"{arm}/{s}", {})
            cur = "-"; row = None
            for n in a.stages:
                if done_models(arm, n, s) is None:
                    cur = n; row = last_row(newest_run(arm, n, s)); break
            else:
                cur = "all done"; row = last_row(newest_run(arm, a.stages[-1], s))
            if row:
                k, r = row
                g = lambda x: float(r.get(x, "nan"))
                L.append(f"| {arm} | {s} | {cur} | {k}/{upd_per_stage} | {int(g('lag_warmup'))} | "
                         f"{g('lag_J_sep'):.3f} ({g('lag_budget_sep'):.3f}, {g('lag_lambda_sep'):.2f}) | "
                         f"{g('lag_J_corr'):.2f} ({g('lag_budget_corr'):.2f}, {g('lag_lambda_corr'):.2f}) | "
                         f"{g('lag_fallback_rate'):.3f} | {g('lag_kl'):.4f} | {st.get('state', '?')} |")
            else:
                L.append(f"| {arm} | {s} | {cur} | 0/{upd_per_stage} | | | | | | {st.get('state', '?')} |")
    L += ["", f"Failures: {json.dumps(state.get('failures', {})) or 'none'}", "", extra]
    (OUT / "status.md").write_text("\n".join(L) + "\n")


def status_loop(a, state, stop):
    while not stop.wait(a.status_hours * 3600):
        write_status(a, state, state["phase"])
        git_push([str(OUT / "status.md")], f"Status: constrained training ({state['phase']})", a)


# ---------------------------------------------------------------- 1. train
def train_one(a, arm, seed, state):
    key = f"{arm}/{seed}"
    cmd = (f"{PY} -u train_lag.py --setup {arm} --seed {seed} --stages {','.join(map(str, a.stages))} "
           f"--steps {a.steps} --threads {a.threads} --episode_length {a.episode_length} --ppo_epoch {a.ppo_epoch} --cpu"
           + (f" --start {a.start}" if a.start else "") + (" --extra " + a.extra if a.extra else ""))
    logf = f"holiday_train_{arm}_seed{seed}.log"
    for attempt in (1, 2):
        state["runs"][key] = {"state": f"training (attempt {attempt})"}
        log(f"{key}: attempt {attempt}")
        if sh(cmd, logf) == 0:
            state["runs"][key] = {"state": "trained"}
            return True
        state.setdefault("failures", {})[key] = f"attempt {attempt} failed at {time.strftime('%d %b %H:%M')} (see {logf})"
        log(f"{key}: attempt {attempt} FAILED")
    state["runs"][key] = {"state": "abandoned"}
    return False


# ---------------------------------------------------------------- 3. export
def export(a, arm, seed):
    dst = WOUT / arm / f"seed{seed}"
    if (dst / "actor.pt").exists():
        return dst
    final = done_models(arm, a.stages[-1], seed)
    if final is None:
        return None
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("actor.pt", "critic.pt", "lag_state.pt"):
        if (final / f).exists():
            shutil.copy2(final / f, dst / f)
    shutil.copy2(Path(a.model_dir) / "config.yaml", dst / "config.yaml")
    for n in a.stages:
        m = done_models(arm, n, seed)
        if m is None:
            continue
        if (m.parent / "lag_log.csv").exists():
            shutil.copy2(m.parent / "lag_log.csv", dst / f"lag_log_N{n}.csv")
        if n != a.stages[-1]:
            (dst / f"stage_N{n}").mkdir(exist_ok=True)
            shutil.copy2(m / "actor.pt", dst / f"stage_N{n}" / "actor.pt")
    return dst


# ---------------------------------------------------------------- 4. evaluation queue
def eval_jobs(a, policies):
    jobs = []

    def add(md, slug, tag, lay, n, s, filt, eps=None):
        E = eps or (30 if n <= 60 else 20)
        jobs.append((n * eplen(lay, n) * E, f"{PY} -u onpolicy/scripts/eval_mpe.py "
                     + BASE.format(md=md, sc=SC[lay], n=n, L=eplen(lay, n), E=E, s=s, t=tag, slug=slug) + (FINAL if filt else ""),
                     tag, slug, lay, n, s))
    for arm, seed, md in policies:
        sl = SLUG[arm]
        for lay, n in CORE:
            add(md, sl, "LGA_tanh", lay, n, seed, False); add(md, sl, "LGF_tanh", lay, n, seed, True)
        for lay, n in NEW:
            add(md, sl, "NS_LGA_tanh", lay, n, seed, False); add(md, sl, "NS_LGF_tanh", lay, n, seed, True)
        for lay, n in DENSE:
            add(md, sl, "NS_LGD_tanh" if lay in ("vertiport", "ladder") else "LGD_tanh", lay, n, seed, True)
    if a.references:
        md = a.model_dir
        for s in a.seeds:
            if s == 4 or s > 3:                    # seeds 1-3 of the original exist already for the core cases
                for lay, n in CORE:
                    add(md, "original", "REFA_tanh", lay, n, s, False); add(md, "original", "REFF_tanh", lay, n, s, True)
                for lay, n in NEW:
                    add(md, "original", "NS_REFF_tanh", lay, n, s, True)
            for lay, n in NEW:                     # original alone on the new layouts, every seed (missing so far)
                add(md, "original", "NS_REFA_tanh", lay, n, s, False)
            for lay, n in DENSE:                   # high density at 20 episodes, every seed
                add(md, "original", "NS_REFD_tanh" if lay in ("vertiport", "ladder") else "REFD_tanh", lay, n, s, True)
    if a.mini:                                     # local test only: two tiny runs per policy
        keep = {("merge", 10), ("vertiport", 30)}
        jobs = [j for j in jobs if (j[4], j[5]) in keep and j[2] in ("LGA_tanh", "NS_LGF_tanh")]
        jobs = [(c, cmd.replace("--render_episodes 30", "--render_episodes 1").replace("--episode_length 300", "--episode_length 30")
                 .replace("--episode_length 450", "--episode_length 30"), *rest) for c, cmd, *rest in jobs]
    jobs.sort(key=lambda j: -j[0])
    return jobs


def result_exists(tag, slug, lay, n, s):
    key = "merge" if lay == "merge" else SC[lay]          # eval_mpe files plain merge as "..._mergeN<n>"
    pat = f"{tag}_{slug}_*_{key}N{n}_*seed{s}_*.txt"
    return any("perep" not in f for f in glob.glob(pat) + glob.glob(str(OUT / pat)))


def run_eval(a, policies, state):
    jobs = [j for j in eval_jobs(a, policies) if not result_exists(*j[2:])]
    log(f"evaluation: {len(jobs)} runs to do")
    state["phase"] = f"evaluation ({len(jobs)} runs)"
    done = [0]

    def one(j):
        rc = sh(j[1], "holiday_eval.log")
        done[0] += 1
        if done[0] % 25 == 0:
            log(f"evaluation: {done[0]}/{len(jobs)} finished")
        return rc
    with ThreadPoolExecutor(a.eval_workers) as ex:
        codes = list(ex.map(one, jobs))
    bad = sum(c != 0 for c in codes)
    if bad:
        state.setdefault("failures", {})["evaluation"] = f"{bad} evaluation runs failed (see holiday_eval.log)"
    return bad


def file_results():
    OUT.mkdir(parents=True, exist_ok=True)
    moved = 0
    for pre in ("LGA_tanh_", "LGF_tanh_", "LGD_tanh_", "NS_LGA_tanh_", "NS_LGF_tanh_", "NS_LGD_tanh_",
                "REFA_tanh_", "REFF_tanh_", "REFD_tanh_", "NS_REFA_tanh_", "NS_REFF_tanh_", "NS_REFD_tanh_"):
        for f in glob.glob(pre + "*"):
            shutil.move(f, OUT / f); moved += 1
    return moved


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arms", default="lag_final,lag_orig")
    p.add_argument("--seeds", default="1,2,3,4")
    p.add_argument("--stages", default="10,15,20")
    p.add_argument("--steps", type=float, default=1e6, help="env steps per stage")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--episode_length", type=int, default=200)
    p.add_argument("--ppo_epoch", type=int, default=4)
    p.add_argument("--start", default="", help="weights for stage 1 (default: train_lag.py's, i.e. model_weights)")
    p.add_argument("--model_dir", default="model_weights")
    p.add_argument("--extra", default="", help="passed to train_lag.py --extra")
    p.add_argument("--eval_workers", type=int, default=16)
    p.add_argument("--status_hours", type=float, default=6.0)
    p.add_argument("--no_references", dest="references", action="store_false")
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--push", action="store_true")
    p.add_argument("--mini", action="store_true", help="local end-to-end test: tiny evaluation queue")
    a = p.parse_args()
    a.arms = a.arms.split(","); a.seeds = [int(s) for s in a.seeds.split(",")]; a.stages = [int(n) for n in a.stages.split(",")]
    if not Path("train_lag.py").exists():
        sys.exit("run from the repo root")
    state = {"start": time.strftime("%a %d %b %H:%M"), "phase": "training", "runs": {}, "failures": {}}
    write_status(a, state, "training (starting)")
    git_push([str(OUT / "status.md")], "Status: constrained training started", a)
    stop = threading.Event()
    threading.Thread(target=status_loop, args=(a, state, stop), daemon=True).start()

    if not a.skip_train:
        with ThreadPoolExecutor(len(a.arms) * len(a.seeds)) as ex:
            futs = [ex.submit(train_one, a, arm, s, state) for arm in a.arms for s in a.seeds]
            ok = [f.result() for f in futs]
        log(f"training finished: {sum(ok)}/{len(ok)} runs complete")

    policies = []
    for arm in a.arms:
        for s in a.seeds:
            d = export(a, arm, s)
            if d is not None:
                policies.append((arm, s, str(d)))
    log(f"exported {len(policies)} policies")
    write_status(a, state, "training done, evaluation starting")
    git_push([str(OUT / "status.md"), str(WOUT)], f"Results: constrained training weights ({len(policies)} policies)", a)

    bad = run_eval(a, policies, state)
    n = file_results()
    state["phase"] = "DONE"
    write_status(a, state, "DONE", extra=f"Evaluation: {n} result files filed in {OUT}; failed evaluation runs: {bad}.")
    stop.set()
    git_push([str(OUT)], f"Results: constrained training E1 + E2 + high density ({n} files)", a)
    log("DONE")


if __name__ == "__main__":
    main()
