# Constrained training - status (Wed 25 Nov 2026 14:40 server time)

Phase: **training**. Started Tue 24 Nov 02:40. Stages [10, 15, 20], 1000000 steps per stage (~312 updates), 16 workers, episodes 200 s, PPO epochs 4.

| Arm | Seed | Stage | Updates in stage | Warm-up | J sep (budget, lambda) | J corr (budget, lambda) | Fallback | KL | State |
|---|---|---|---|---|---|---|---|---|---|
| lag_orig2 | 1 | 20 | 66/312 | 0 | 0.855 (0.384, 3.07) | 0.14 (0.18, 2.05) | 0.000 | 0.1166 | training (attempt 1) |
| lag_orig2 | 2 | 20 | 73/312 | 0 | 0.653 (0.690, 1.06) | 0.03 (0.22, 2.20) | 0.000 | 0.0586 | training (attempt 1) |
| lag_orig2 | 3 | 20 | 71/312 | 0 | 0.441 (0.474, 1.87) | 0.06 (0.10, 2.98) | 0.000 | 0.1032 | training (attempt 1) |
| lag_orig2 | 4 | 20 | 81/312 | 0 | 0.202 (0.367, 4.54) | 0.24 (0.10, 3.00) | 0.000 | 0.1248 | training (attempt 1) |
| lag_teach | 1 | 15 | 0/312 | | | | | | abandoned |
| lag_teach | 2 | 15 | 0/312 | | | | | | abandoned |
| lag_teach | 3 | 15 | 0/312 | | | | | | abandoned |
| lag_teach | 4 | 15 | 0/312 | | | | | | abandoned |

Failures: {"lag_teach/3": "attempt 2 failed at 24 Nov 16:32 (see holiday_train_lag_teach_seed3.log)", "lag_teach/1": "attempt 2 failed at 24 Nov 16:32 (see holiday_train_lag_teach_seed1.log)", "lag_teach/4": "attempt 2 failed at 24 Nov 16:32 (see holiday_train_lag_teach_seed4.log)", "lag_teach/2": "attempt 2 failed at 24 Nov 16:32 (see holiday_train_lag_teach_seed2.log)"}


