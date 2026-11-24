# Constrained training - status (Tue 24 Nov 2026 20:40 server time)

Phase: **training**. Started Tue 24 Nov 02:40. Stages [10, 15, 20], 1000000 steps per stage (~312 updates), 16 workers, episodes 200 s, PPO epochs 4.

| Arm | Seed | Stage | Updates in stage | Warm-up | J sep (budget, lambda) | J corr (budget, lambda) | Fallback | KL | State |
|---|---|---|---|---|---|---|---|---|---|
| lag_orig2 | 1 | 15 | 164/312 | 0 | 0.715 (1.033, 0.57) | 0.02 (0.36, 0.51) | 0.000 | 0.1003 | training (attempt 1) |
| lag_orig2 | 2 | 15 | 167/312 | 0 | 0.565 (1.171, 0.73) | 0.68 (0.29, 2.51) | 0.000 | 0.1213 | training (attempt 1) |
| lag_orig2 | 3 | 15 | 168/312 | 0 | 0.958 (1.244, 0.00) | 0.06 (0.87, 0.00) | 0.000 | 0.0751 | training (attempt 1) |
| lag_orig2 | 4 | 15 | 175/312 | 0 | 0.785 (1.117, 0.41) | 0.70 (0.12, 3.00) | 0.000 | 0.0878 | training (attempt 1) |
| lag_teach | 1 | 15 | 0/312 | | | | | | abandoned |
| lag_teach | 2 | 15 | 0/312 | | | | | | abandoned |
| lag_teach | 3 | 15 | 0/312 | | | | | | abandoned |
| lag_teach | 4 | 15 | 0/312 | | | | | | abandoned |

Failures: {"lag_teach/3": "attempt 2 failed at 24 Nov 16:32 (see holiday_train_lag_teach_seed3.log)", "lag_teach/1": "attempt 2 failed at 24 Nov 16:32 (see holiday_train_lag_teach_seed1.log)", "lag_teach/4": "attempt 2 failed at 24 Nov 16:32 (see holiday_train_lag_teach_seed4.log)", "lag_teach/2": "attempt 2 failed at 24 Nov 16:32 (see holiday_train_lag_teach_seed2.log)"}


