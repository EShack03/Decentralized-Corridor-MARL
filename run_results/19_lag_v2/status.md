# Constrained training - status (Thu 26 Nov 2026 14:40 server time)

Phase: **training**. Started Tue 24 Nov 02:40. Stages [10, 15, 20], 1000000 steps per stage (~312 updates), 16 workers, episodes 200 s, PPO epochs 4.

| Arm | Seed | Stage | Updates in stage | Warm-up | J sep (budget, lambda) | J corr (budget, lambda) | Fallback | KL | State |
|---|---|---|---|---|---|---|---|---|---|
| lag_orig2 | 1 | 20 | 302/312 | 0 | 0.236 (0.384, 0.00) | 0.01 (0.18, 0.00) | 0.000 | 0.0753 | training (attempt 1) |
| lag_orig2 | 2 | 20 | 286/312 | 0 | 0.430 (0.690, 0.00) | 0.01 (0.22, 0.00) | 0.000 | 0.0529 | training (attempt 1) |
| lag_orig2 | 3 | 20 | 297/312 | 0 | 0.591 (0.474, 0.43) | 0.00 (0.10, 1.43) | 0.000 | 0.0763 | training (attempt 1) |
| lag_orig2 | 4 | all done | 312/312 | 0 | 0.309 (0.367, 0.00) | 0.00 (0.10, 0.00) | 0.000 | 0.0688 | trained |
| lag_teach | 1 | 15 | 0/312 | | | | | | abandoned |
| lag_teach | 2 | 15 | 0/312 | | | | | | abandoned |
| lag_teach | 3 | 15 | 0/312 | | | | | | abandoned |
| lag_teach | 4 | 15 | 0/312 | | | | | | abandoned |

Failures: {"lag_teach/3": "attempt 2 failed at 24 Nov 16:32 (see holiday_train_lag_teach_seed3.log)", "lag_teach/1": "attempt 2 failed at 24 Nov 16:32 (see holiday_train_lag_teach_seed1.log)", "lag_teach/4": "attempt 2 failed at 24 Nov 16:32 (see holiday_train_lag_teach_seed4.log)", "lag_teach/2": "attempt 2 failed at 24 Nov 16:32 (see holiday_train_lag_teach_seed2.log)"}


