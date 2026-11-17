"""
[ES] added (17 Nov 2026) - distillation driver: DAgger rounds = collect (CPU, parallel) -> train (CPU, seeds in parallel).

Round r (r = 0..R-1), for each training seed s:
  collect: the 4 original layouts x 10/20/30/40 aircraft x --jobs episodes-files, the teacher flying with
           probability beta_r and the student of round r-1 otherwise (round 0: teacher only);
  train:   on every shard of seed s from rounds 0..r, starting from the round r-1 student
           (round 0: the original weights) -> model_weights_distill/seed{s}/r{r}.
After each round every new student also flies alone (beta 0, no filter) on each layout x count ("check
flights", data_root/seed{s}/check_r{r}; never used for training) to track its own success and losses.
The original policy alone is flown once on the same check seeds as the reference. Environment seeds are
10000*s + 100*r + 10*j + count index for collection and 10000*s + 50 + count index for every check flight (the
same each round, so rounds are compared on identical starts); none coincides with the evaluation seeds (1-3).
Re-running the same command resumes: shards and weights that already exist are skipped.

Usage (repo root, venv active):
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python onpolicy/scripts/distill_run.py --seeds 1 2 3 --workers 16
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

LAYOUTS = {"single": "working_three_phase_graph_hidegoals", "seq": "three_phase_graph_sequential_hidegoals",
           "merge": "three_phase_graph_merge", "mergeseq": "three_phase_graph_merge_sequential"}
COUNTS = (10, 20, 30, 40)
LEN = {"mergeseq": {10: 300, 20: 350, 30: 450, 40: 550}}
LEN_DEFAULT = {10: 300, 20: 300, 30: 450, 40: 550}
BASE = ("--model_dir {md} --env_name GraphMPE --algorithm_name rmappo --dynamics_type air_taxi --num_agents {n} "
        "--num_landmarks {n} --world_size 5.0 --episode_length {L} --render_episodes {E} --seed {seed} "
        "--formation_type point --total_actions 9 --zeroshift 10 --use_dones False --ifi 0.0 --use_ReLU "
        "--scenario_name {sc} --policy_slug original")
FILTER = (" --cbf_eval --cbf_barrier cbf_stage1_barrier.pt --cbf_alpha 1.0 --cbf_horizon 3 --cbf_fallback least_unsafe"
          " --cbf_filter --cbf_neighbour_model route_map --cbf_ignore_parked --cbf_no_barrier --cbf_sep_scale 1.2"
          " --cbf_route_variant fix --cbf_exact_dynamics --cbf_sense_ground_vel --cbf_sense_pos_sigma 0"
          " --cbf_sense_vel_sigma 0 --cbf_sense_delay 0 --cbf_sense_kalman --cbf_sense_compensate")


def run(cmd, log):
    with open(log, "w") as f:
        return subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT,
                              env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}).returncode


def make_jobs(a, d, beta, student, seedbase, n_jobs, episodes):
    os.makedirs(d, exist_ok=True)
    jobs = []
    for li, lay in enumerate(a.layouts):
        for ci, n in enumerate(a.counts):
            for j in range(n_jobs):
                out = os.path.join(d, f"{lay}_N{n}_j{j}.npz")
                if os.path.exists(out):
                    continue
                L = LEN.get(lay, LEN_DEFAULT)[n]
                cmd = (f"{a.python} -u onpolicy/scripts/distill_collect.py "
                       + BASE.format(md=a.model_dir, n=n, L=L, E=episodes, seed=seedbase + 10 * j + ci, sc=LAYOUTS[lay]) + FILTER
                       + f" --distill_out {out} --distill_beta {beta} --distill_keep_frac {a.keep_frac} --distill_keep_blocked {a.keep_blocked}"
                       + (f" --distill_student {student}" if student else ""))
                jobs.append((n * L, cmd, out[:-4] + ".log"))
    return jobs


def pool(a, jobs, what):
    jobs.sort(key=lambda x: -x[0])                 # longest first
    t0 = time.time()
    print(f"[distill-run] {what}: {len(jobs)} jobs", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        codes = list(ex.map(lambda j: run(j[1], j[2]), jobs))
    bad = [j[2] for j, c in zip(jobs, codes) if c != 0]
    print(f"[distill-run] {what} done in {time.time() - t0:.0f} s; failed jobs: {len(bad)}", flush=True)
    if bad:
        print("\n".join(bad)); sys.exit(1)


def flown(d, beta):
    """Per-folder flight summary from the shards' episode records."""
    ep = np.concatenate([np.load(f)["episodes"] for f in sorted(glob.glob(os.path.join(d, "*.npz")))])
    fly = max(ep[:, 2].sum(), 1)
    return dict(beta=beta, episodes=len(ep), success=round(float(ep[:, 0].mean()), 5),
                loss_pair_seconds_per_ep=round(float(ep[:, 1].mean()), 3),
                changed_rate=round(float(ep[:, 3].sum() / fly), 5), removed_only_rate=round(float(ep[:, 8].sum() / fly), 5),
                no_safe=float(ep[:, 5].sum()), min_sep_m=round(1000 * float(ep[:, 7].min()), 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--betas", type=str, default="1,0.5,0.25,0.1,0", help="teacher share per round (length = rounds)")
    p.add_argument("--jobs", type=int, default=4, help="collection jobs per layout x count x seed x round")
    p.add_argument("--episodes", type=int, default=3, help="episodes per collection job")
    p.add_argument("--check_episodes", type=int, default=2, help="student-alone check episodes per layout x count after each round (0 = none)")
    p.add_argument("--counts", type=int, nargs="+", default=list(COUNTS))
    p.add_argument("--layouts", type=str, nargs="+", default=list(LAYOUTS))
    p.add_argument("--keep_frac", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--w_int", type=float, default=5.0)
    p.add_argument("--ce", type=float, default=0.0)
    p.add_argument("--keep_blocked", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cpu", help="cpu (default: no GPU notice needed) | cuda | auto")
    p.add_argument("--model_dir", type=str, default="model_weights")
    p.add_argument("--data_root", type=str, default="distill_data")
    p.add_argument("--weights_root", type=str, default="model_weights_distill")
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument("--max_samples", type=int, default=0, help="per-shard cap in training (quick tests only)")
    a = p.parse_args()
    betas = [float(b) for b in a.betas.split(",")]
    os.makedirs(a.data_root, exist_ok=True); os.makedirs(a.weights_root, exist_ok=True)
    summary_path = os.path.join(a.weights_root, "rounds_summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}

    # reference: the ORIGINAL policy alone on the same check-flight seeds (paired with every round's check flights)
    if a.check_episodes > 0:
        jobs = []
        for s in a.seeds:
            jobs += make_jobs(a, os.path.join(a.data_root, f"seed{s}", "check_original"), 0.0, "", 10000 * s + 50, 1, a.check_episodes)
        pool(a, jobs, "original policy alone, check flights")
        for s in a.seeds:
            summary[f"seed{s}_check_original_alone"] = flown(os.path.join(a.data_root, f"seed{s}", "check_original"), 0.0)
            print(f"[distill-run] seed {s} original alone: {summary[f'seed{s}_check_original_alone']}", flush=True)

    for r, beta in enumerate(betas):
        # ---------------- collect (all seeds in one pool) ----------------
        jobs = []
        for s in a.seeds:
            prev = os.path.join(a.weights_root, f"seed{s}", f"r{r - 1}") if r > 0 else ""
            jobs += make_jobs(a, os.path.join(a.data_root, f"seed{s}", f"r{r}"), beta, prev, 10000 * s + 100 * r, a.jobs, a.episodes)
        pool(a, jobs, f"round {r} (teacher share {beta}) collection")

        # ---------------- train (the seeds in parallel, one process each; CPU by default) ----------------
        todo = []
        for s in a.seeds:
            save = os.path.join(a.weights_root, f"seed{s}", f"r{r}")
            summary[f"seed{s}_r{r}_collected"] = flown(os.path.join(a.data_root, f"seed{s}", f"r{r}"), beta)
            print(f"[distill-run] seed {s} round {r} flown data: {summary[f'seed{s}_r{r}_collected']}", flush=True)
            if os.path.exists(os.path.join(save, "actor.pt")):
                continue
            init = os.path.join(a.weights_root, f"seed{s}", f"r{r - 1}") if r > 0 else a.model_dir
            cmd = (f"{a.python} -u onpolicy/scripts/distill_train.py "
                   + BASE.format(md=a.model_dir, n=10, L=300, E=1, seed=s, sc=LAYOUTS["merge"])
                   + f" --distill_data '{os.path.join(a.data_root, f'seed{s}', 'r*', '*.npz')}' --distill_init {init}"
                   + f" --distill_save {save} --distill_epochs {a.epochs} --distill_lr {a.lr} --distill_w_int {a.w_int}"
                   + f" --distill_ce {a.ce} --distill_device {a.device} --distill_max_samples {a.max_samples}")
            os.makedirs(save, exist_ok=True)
            todo.append((s, save, cmd))
        t0 = time.time()
        with ThreadPoolExecutor(max(1, len(todo))) as ex:
            codes = list(ex.map(lambda x: run(x[2], os.path.join(x[1], "train.log")), todo))
        for (s, save, _), c in zip(todo, codes):
            if c != 0:
                print(f"[distill-run] training failed: {save}/train.log"); sys.exit(1)
            summary[f"seed{s}_r{r}_trained"] = json.load(open(os.path.join(save, "distill_log.json")))["log"][-1]
            print(f"[distill-run] seed {s} round {r} trained ({time.time() - t0:.0f} s for all seeds): {summary[f'seed{s}_r{r}_trained']}", flush=True)
        # ---------------- check flights: each new student flies alone (beta 0), never used for training ----------------
        if a.check_episodes > 0:
            jobs = []
            for s in a.seeds:
                jobs += make_jobs(a, os.path.join(a.data_root, f"seed{s}", f"check_r{r}"), 0.0,
                                  os.path.join(a.weights_root, f"seed{s}", f"r{r}"), 10000 * s + 50, 1, a.check_episodes)
            pool(a, jobs, f"round {r} check flights")
            for s in a.seeds:
                summary[f"seed{s}_r{r}_check_student_alone"] = flown(os.path.join(a.data_root, f"seed{s}", f"check_r{r}"), 0.0)
                print(f"[distill-run] seed {s} round {r} student alone: {summary[f'seed{s}_r{r}_check_student_alone']}", flush=True)
        json.dump(summary, open(summary_path, "w"), indent=1)
    print("[distill-run] DONE", flush=True)


if __name__ == "__main__":
    main()
