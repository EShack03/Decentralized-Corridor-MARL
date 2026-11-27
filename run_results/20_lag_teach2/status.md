# Constrained training - status (Fri 27 Nov 2026 18:38 server time)

Phase: **DONE**. Started Tue 24 Nov 16:54. Stages [10, 15, 20], 1000000 steps per stage (~312 updates), 16 workers, episodes 200 s, PPO epochs 4.

| Arm | Seed | Stage | Updates in stage | Warm-up | J sep (budget, lambda) | J corr (budget, lambda) | Fallback | KL | State |
|---|---|---|---|---|---|---|---|---|---|
| lag_teach2 | 1 | all done | 312/312 | 0 | 0.622 (0.918, 0.00) | 0.03 (0.30, 2.32) | 0.000 | 0.0249 | trained |
| lag_teach2 | 2 | all done | 312/312 | 0 | 1.220 (1.057, 0.03) | 0.20 (0.20, 2.58) | 0.000 | 0.0357 | trained |
| lag_teach2 | 3 | all done | 312/312 | 0 | 0.673 (0.651, 0.02) | 0.82 (0.49, 0.87) | 0.000 | 0.0317 | trained |
| lag_teach2 | 4 | all done | 312/312 | 0 | 0.689 (0.870, 0.00) | 0.99 (0.16, 3.00) | 0.000 | 0.0284 | trained |

Failures: {}

Evaluation: 544 result files filed in run_results/20_lag_teach2; failed evaluation runs: 0.
