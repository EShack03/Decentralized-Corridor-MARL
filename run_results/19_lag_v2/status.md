# Constrained training - status (Wed 25 Nov 2026 02:40 server time)

Phase: **training**. Started Tue 24 Nov 02:40. Stages [10, 15, 20], 1000000 steps per stage (~312 updates), 16 workers, episodes 200 s, PPO epochs 4.

| Arm | Seed | Stage | Updates in stage | Warm-up | J sep (budget, lambda) | J corr (budget, lambda) | Fallback | KL | State |
|---|---|---|---|---|---|---|---|---|---|
| lag_orig2 | 1 | 15 | 246/312 | 0 | 0.767 (1.033, 0.00) | 0.07 (0.36, 0.00) | 0.000 | 0.1004 | training (attempt 1) |
| lag_orig2 | 2 | 15 | 252/312 | 0 | 1.392 (1.171, 0.14) | 0.45 (0.29, 0.42) | 0.000 | 0.0847 | training (attempt 1) |
| lag_orig2 | 3 | 15 | 253/312 | 0 | 1.402 (1.244, 0.07) | 0.06 (0.87, 0.00) | 0.000 | 0.0885 | training (attempt 1) |
| lag_orig2 | 4 | 15 | 259/312 | 0 | 0.794 (1.117, 0.00) | 0.52 (0.12, 1.45) | 0.000 | 0.0728 | training (attempt 1) |
| lag_teach | 1 | 15 | 0/312 | | | | | | abandoned |
| lag_teach | 2 | 15 | 0/312 | | | | | | abandoned |
| lag_teach | 3 | 15 | 0/312 | | | | | | abandoned |
| lag_teach | 4 | 15 | 0/312 | | | | | | abandoned |

Failures: {"lag_teach/3": "attempt 2 failed at 24 Nov 16:32 (see holiday_train_lag_teach_seed3.log)", "lag_teach/1": "attempt 2 failed at 24 Nov 16:32 (see holiday_train_lag_teach_seed1.log)", "lag_teach/4": "attempt 2 failed at 24 Nov 16:32 (see holiday_train_lag_teach_seed4.log)", "lag_teach/2": "attempt 2 failed at 24 Nov 16:32 (see holiday_train_lag_teach_seed2.log)"}


