# Constrained training - status (Wed 25 Nov 2026 20:40 server time)

Phase: **training**. Started Tue 24 Nov 02:40. Stages [10, 15, 20], 1000000 steps per stage (~312 updates), 16 workers, episodes 200 s, PPO epochs 4.

| Arm | Seed | Stage | Updates in stage | Warm-up | J sep (budget, lambda) | J corr (budget, lambda) | Fallback | KL | State |
|---|---|---|---|---|---|---|---|---|---|
| lag_orig2 | 1 | 20 | 118/312 | 0 | 0.272 (0.384, 1.92) | 0.48 (0.18, 3.00) | 0.000 | 0.1192 | training (attempt 1) |
| lag_orig2 | 2 | 20 | 121/312 | 0 | 0.686 (0.690, 0.37) | 0.03 (0.22, 1.30) | 0.000 | 0.0539 | training (attempt 1) |
| lag_orig2 | 3 | 20 | 122/312 | 0 | 0.605 (0.474, 1.44) | 0.01 (0.10, 1.52) | 0.000 | 0.0936 | training (attempt 1) |
| lag_orig2 | 4 | 20 | 134/312 | 0 | 0.287 (0.367, 5.42) | 0.88 (0.10, 3.00) | 0.000 | 0.1228 | training (attempt 1) |
| lag_teach | 1 | 15 | 0/312 | | | | | | abandoned |
| lag_teach | 2 | 15 | 0/312 | | | | | | abandoned |
| lag_teach | 3 | 15 | 0/312 | | | | | | abandoned |
| lag_teach | 4 | 15 | 0/312 | | | | | | abandoned |

Failures: {"lag_teach/3": "attempt 2 failed at 24 Nov 16:32 (see holiday_train_lag_teach_seed3.log)", "lag_teach/1": "attempt 2 failed at 24 Nov 16:32 (see holiday_train_lag_teach_seed1.log)", "lag_teach/4": "attempt 2 failed at 24 Nov 16:32 (see holiday_train_lag_teach_seed4.log)", "lag_teach/2": "attempt 2 failed at 24 Nov 16:32 (see holiday_train_lag_teach_seed2.log)"}


