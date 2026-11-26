# Constrained training - status (Thu 26 Nov 2026 02:40 server time)

Phase: **training**. Started Tue 24 Nov 02:40. Stages [10, 15, 20], 1000000 steps per stage (~312 updates), 16 workers, episodes 200 s, PPO epochs 4.

| Arm | Seed | Stage | Updates in stage | Warm-up | J sep (budget, lambda) | J corr (budget, lambda) | Fallback | KL | State |
|---|---|---|---|---|---|---|---|---|---|
| lag_orig2 | 1 | 20 | 181/312 | 0 | 0.261 (0.384, 0.65) | 0.01 (0.18, 1.39) | 0.000 | 0.0962 | training (attempt 1) |
| lag_orig2 | 2 | 20 | 175/312 | 0 | 0.569 (0.690, 0.00) | 0.03 (0.22, 0.00) | 0.000 | 0.0795 | training (attempt 1) |
| lag_orig2 | 3 | 20 | 180/312 | 0 | 0.530 (0.474, 0.72) | 0.04 (0.10, 0.54) | 0.000 | 0.0817 | training (attempt 1) |
| lag_orig2 | 4 | 20 | 198/312 | 0 | 0.144 (0.367, 2.42) | 1.95 (0.10, 3.00) | 0.000 | 0.1211 | training (attempt 1) |
| lag_teach | 1 | 15 | 0/312 | | | | | | abandoned |
| lag_teach | 2 | 15 | 0/312 | | | | | | abandoned |
| lag_teach | 3 | 15 | 0/312 | | | | | | abandoned |
| lag_teach | 4 | 15 | 0/312 | | | | | | abandoned |

Failures: {"lag_teach/3": "attempt 2 failed at 24 Nov 16:32 (see holiday_train_lag_teach_seed3.log)", "lag_teach/1": "attempt 2 failed at 24 Nov 16:32 (see holiday_train_lag_teach_seed1.log)", "lag_teach/4": "attempt 2 failed at 24 Nov 16:32 (see holiday_train_lag_teach_seed4.log)", "lag_teach/2": "attempt 2 failed at 24 Nov 16:32 (see holiday_train_lag_teach_seed2.log)"}


