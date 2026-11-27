# Constrained training - status (Fri 27 Nov 2026 05:48 server time)

Phase: **DONE**. Started Tue 24 Nov 02:40. Stages [10, 15, 20], 1000000 steps per stage (~312 updates), 16 workers, episodes 200 s, PPO epochs 4.

| Arm | Seed | Stage | Updates in stage | Warm-up | J sep (budget, lambda) | J corr (budget, lambda) | Fallback | KL | State |
|---|---|---|---|---|---|---|---|---|---|
| lag_orig2 | 1 | all done | 312/312 | 0 | 0.298 (0.384, 0.00) | 0.02 (0.18, 0.00) | 0.000 | 0.0815 | trained |
| lag_orig2 | 2 | all done | 312/312 | 0 | 0.703 (0.690, 0.00) | 0.21 (0.22, 0.05) | 0.000 | 0.0646 | trained |
| lag_orig2 | 3 | all done | 312/312 | 0 | 0.755 (0.474, 0.60) | 0.81 (0.10, 2.61) | 0.000 | 0.0934 | trained |
| lag_orig2 | 4 | all done | 312/312 | 0 | 0.309 (0.367, 0.00) | 0.00 (0.10, 0.00) | 0.000 | 0.0688 | trained |
| lag_teach | 1 | 15 | 0/312 | | | | | | abandoned |
| lag_teach | 2 | 15 | 0/312 | | | | | | abandoned |
| lag_teach | 3 | 15 | 0/312 | | | | | | abandoned |
| lag_teach | 4 | 15 | 0/312 | | | | | | abandoned |

Failures: {"lag_teach/3": "attempt 2 failed at 24 Nov 16:32 (see holiday_train_lag_teach_seed3.log)", "lag_teach/1": "attempt 2 failed at 24 Nov 16:32 (see holiday_train_lag_teach_seed1.log)", "lag_teach/4": "attempt 2 failed at 24 Nov 16:32 (see holiday_train_lag_teach_seed4.log)", "lag_teach/2": "attempt 2 failed at 24 Nov 16:32 (see holiday_train_lag_teach_seed2.log)"}

Evaluation: 544 result files filed in run_results/19_lag_v2; failed evaluation runs: 0.
