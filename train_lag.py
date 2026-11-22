#!/usr/bin/env python3
# ============================================================================
# train_lag.py - run one constrained fine-tuning set-up end to end
# ----------------------------------------------------------------------------
# [ES] added (constrained training, Nov 2026). Builds and runs the
# onpolicy/scripts/train_mpe_lag.py commands for one set-up and one training seed,
# through a density curriculum (default 10 -> 15 -> 20 aircraft). Each stage
# starts from the previous stage's weights; stage 1 starts from the original
# policy (model_weights/). Every stage mixes the four layouts across the
# rollout workers. The original training flags (rewards, network, PopArt) are
# passed unchanged so the restored actor and critic load exactly.
#
# Set-ups (--setup):
#   penalty        standard MAPPO, fixed cost penalties in the reward, filter on
#   lagrangian     MAPPO-Lagrangian with PID multipliers, filter on   (main)
#   lag_nofilter   MAPPO-Lagrangian with PID multipliers, filter off
#
#   python train_lag.py --setup lagrangian --seed 1 --dry_run   # print commands
#   python train_lag.py --setup lagrangian --seed 1             # run all stages
#   python train_lag.py --setup lagrangian --seed 1 --smoke     # ~2 min CPU check
#
# Finished stages are skipped on a rerun (a DONE file in the stage's models
# folder), so a stopped run resumes at the first unfinished stage.
# Output: onpolicy/results/GraphMPE/<scenario>/rmappo/lag_<setup>_N<n>_seed<k>/runK/
#   models/actor.pt, critic.pt, lag_state.pt, DONE      logs/ (tensorboard)
#   lag_log.csv  one row per update: costs, budgets, multipliers, KL, fallback
# ============================================================================
import argparse
import os
import subprocess
import sys
from pathlib import Path

# [ES] 20 / 22 Nov - v2 settings (arm B v2 and arm C), from the arm B v1 results (both failed seeds ended with
# the corridor multiplier pinned at its cap): bounded symmetric PID error and integrator clamping (anti-windup);
# the corridor is a guard, not a target - budget 1.0 x the starting cost (v1: 0.7), floor 0.1 aircraft-seconds
# off the corridor per aircraft per episode (= 0.05 % of a 200 s episode; v1: 0.01), multiplier cap 3 (v1: 10).
# Separation unchanged from v1 (factor 1.0, floor 0.01, cap 10).
V2 = ["--lag_err_clip", "1.0", "--lag_int_clamp", "True", "--lag_budget_factor_corr", "1.0",
      "--lag_budget_floor_corr", "0.1", "--lag_lambda_max_corr", "3.0"]
SETUPS = {
    "penalty":      ["--lag_mode", "penalty", "--lag_filter", "True"],
    "lagrangian":   ["--lag_mode", "lagrangian", "--lag_filter", "True"],
    "lag_nofilter": ["--lag_mode", "lagrangian", "--lag_filter", "False"],
    # [ES] 18 Nov: from the distilled student, no filter anywhere, budget = the start policy's own cost
    "lag_distilled": ["--lag_mode", "lagrangian", "--lag_filter", "False", "--lag_budget_with_filter", "False",
                      "--lag_mask_coef", "0"],
    # [ES] 18 Nov (holiday run), both from the ORIGINAL policy, final filter where a filter is used:
    #   A: final filter in the training loop (deployed with it); budget = original + filter cost
    "lag_final": ["--lag_mode", "lagrangian", "--lag_filter", "True", "--lag_filter_kind", "final",
                  "--lag_mask_coef", "0"],
    #   B: no filter in training; warm-up measures the original + final filter cost as the budget the
    #      policy must reach on its own
    "lag_orig": ["--lag_mode", "lagrangian", "--lag_filter", "False", "--lag_filter_kind", "final",
                 "--lag_budget_with_filter", "True", "--lag_mask_coef", "0"],
    # [ES] 20 Nov - arm C ("teach"), from the ORIGINAL policy, aimed at a policy that flies ALONE as safely as the
    # original + final filter: the final filter is applied to a share of the aircraft that falls stage by stage
    # (--filter_probs, default 0.8 / 0.4 / 0), its mask is computed for every aircraft and a mask-consistency term
    # teaches the policy to put no probability on the moves it blocks, and the Lagrangian budget is the original
    # + final filter cost (warm-up with the filter), applied to the policy's own cost (unfiltered aircraft).
    "lag_teach": ["--lag_mode", "lagrangian", "--lag_filter", "True", "--lag_filter_kind", "final",
                  "--lag_mask_coef", "1.0", "--lag_aux_balanced", "True", "--lag_aux_rows", "8192",
                  *V2],
    # [ES] 20 Nov - arm B v2: arm B (lag_orig) with the v2 multiplier settings, identical to arm C except
    # for the teaching (no filter in the loop, no mask term), so B v2 vs C isolates the filter as teacher.
    "lag_orig2": ["--lag_mode", "lagrangian", "--lag_filter", "False", "--lag_filter_kind", "final",
                  "--lag_budget_with_filter", "True", "--lag_mask_coef", "0", *V2],
}
# share of aircraft with the filter applied, per stage, for set-ups that fade the filter out
FILTER_PROBS = {"lag_teach": "0.8,0.4,0"}
LAYOUTS = ["single", "sequential", "merge", "mergeseq"]
FIRST_SCEN = {"single": "working_three_phase_graph_hidegoals",
              "sequential": "three_phase_graph_sequential_hidegoals",
              "merge": "three_phase_graph_merge",
              "mergeseq": "three_phase_graph_merge_sequential"}

# the original training configuration (model_weights/config.yaml)
ORIGINAL = ["--env_name", "GraphMPE", "--algorithm_name", "rmappo",
            "--dynamics_type", "air_taxi", "--world_size", "5",
            "--use_valuenorm", "--use_popart",          # = PopArt on, ValueNorm off
            "--collision_rew", "20", "--formation_rew", "10", "--goal_rew", "20",
            "--fair_rew", "10", "--fair_wt", "1", "--formation_type", "point",
            "--total_actions", "9", "--zeroshift", "10", "--graph_feat_type", "relative",
            "--use_ReLU",                               # store_false: Tanh, as trained
            "--use_cent_obs", "False", "--use_dones", "False", "--ppo_epoch", "10",
            "--auto_mini_batch_size",
            "--use_wandb"]                               # store_false: turns wandb off


def stage_dir(scen, exp):
    return Path("onpolicy/results/GraphMPE") / scen / "rmappo" / exp


def done_models(scen, exp):
    """models/ of a finished run of this stage, else None."""
    d = stage_dir(scen, exp)
    if not d.is_dir():
        return None
    for run in sorted(d.glob("run*"), key=lambda p: int(p.name[3:] or 0), reverse=True):
        if (run / "models" / "DONE").exists():
            return run / "models"
    return None


def newest_models(scen, exp):
    runs = sorted(stage_dir(scen, exp).glob("run*"), key=lambda p: int(p.name[3:] or 0))
    return runs[-1] / "models" if runs else None


def main():
    ap = argparse.ArgumentParser(description="constrained fine-tuning driver")
    ap.add_argument("--setup", choices=list(SETUPS), required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--stages", default="10,15,20", help="aircraft per stage")
    ap.add_argument("--steps", type=float, default=1.5e6,
                    help="env steps per stage (one step = one env x one second)")
    ap.add_argument("--threads", type=int, default=32, help="rollout workers")
    ap.add_argument("--episode_length", type=int, default=200)
    ap.add_argument("--layouts", default=",".join(LAYOUTS))
    ap.add_argument("--horizon", type=int, default=3, help="training-filter look-ahead (s)")
    ap.add_argument("--lr", default="2e-4", help="actor learning rate (original 5e-4)")
    ap.add_argument("--critic_lr", default="5e-4")
    ap.add_argument("--entropy_coef", default="0.02")
    ap.add_argument("--ppo_epoch", default="10", help="PPO epochs per update (original 10)")
    ap.add_argument("--filter_probs", default="",
                    help="per-stage share of aircraft with the filter applied (default: FILTER_PROBS[setup], else none)")
    ap.add_argument("--start", default="", help="weights for stage 1 (default: model_weights, or model_weights_distill/seed<seed>/r4 for --setup lag_distilled)")
    ap.add_argument("--cpu", action="store_true", help="train on CPU (no GPU)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU check: 4 workers, 40-step episodes, 3 updates")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="anything after --extra is passed to train_mpe_lag.py")
    a = ap.parse_args()
    if not Path("onpolicy/scripts/train_mpe_lag.py").exists():
        sys.exit("run from the repo root")

    if not a.start:
        a.start = f"model_weights_distill/seed{a.seed}/r4" if a.setup == "lag_distilled" else "model_weights"
    stages = [int(n) for n in a.stages.split(",")]
    layouts = a.layouts.split(",")
    threads, eplen, steps, warm = a.threads, a.episode_length, a.steps, 5
    if a.smoke:
        stages, threads, eplen, warm = stages[:1], 4, 40, 1
        steps, a.cpu = 3 * 4 * 40, True
    if threads < 2:
        sys.exit("--threads must be >= 2 (the 1-worker vec env is not supported in training)")
    scen0 = FIRST_SCEN.get(layouts[0], layouts[0])
    py = sys.executable
    prev = a.start
    fp = (a.filter_probs or FILTER_PROBS.get(a.setup, "")).split(",") if (a.filter_probs or a.setup in FILTER_PROBS) else []
    if fp and len(fp) < len(stages):
        sys.exit(f"--filter_probs needs one value per stage ({len(stages)})")
    for si, n in enumerate(stages):
        exp = f"lag_{a.setup}_N{n}_seed{a.seed}" + ("_smoke" if a.smoke else "")
        have = done_models(scen0, exp)
        if have and not a.smoke:
            print(f"[have] stage N={n}: {have}")
            prev = str(have)
            continue
        cmd = [py, "-u", "onpolicy/scripts/train_mpe_lag.py", *ORIGINAL,
               "--scenario_name", scen0, "--num_agents", str(n), "--num_landmarks", str(n),
               "--episode_length", str(eplen), "--n_rollout_threads", str(threads),
               "--num_env_steps", str(int(steps)), "--seed", str(a.seed),
               "--experiment_name", exp, "--model_dir", prev,
               "--lr", a.lr, "--critic_lr", a.critic_lr, "--entropy_coef", a.entropy_coef,
               "--ppo_epoch", a.ppo_epoch,
               # minibatches of ~16k agent-steps, or two halves when a rollout is smaller
               "--target_mini_batch_size", str(min(16384, threads * eplen * n // 2)),
               "--log_interval", "5", "--save_interval", "5",
               "--lag_enable", *SETUPS[a.setup], "--lag_scenarios", ",".join(layouts),
               "--lag_filter_horizon", str(a.horizon), "--lag_warmup", str(warm),
               *(["--lag_filter_prob", fp[si]] if fp else []),
               *a.extra]
        if a.cpu:
            cmd += ["--cuda"]                        # store_false: turns CUDA off
        if a.smoke:
            cmd += ["--num_mini_batch", "1"]
            cmd.remove("--auto_mini_batch_size")
        print(f"\n=== stage N={n}  setup={a.setup}  seed={a.seed}  from {prev}")
        print(" ".join(cmd))
        if a.dry_run:
            prev = f"<models of {exp}>"
            continue
        subprocess.run(cmd, check=True, env=dict(os.environ, OMP_NUM_THREADS="1",
                                                 MKL_NUM_THREADS="1"))
        models = newest_models(scen0, exp)
        (models / "DONE").write_text("ok\n")
        prev = str(models)
        print(f"=== stage N={n} done -> {models}")


if __name__ == "__main__":
    main()
