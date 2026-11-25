# Constrained training - status (Wed 25 Nov 2026 08:40 server time)

Phase: **training**. Started Tue 24 Nov 02:40. Stages [10, 15, 20], 1000000 steps per stage (~312 updates), 16 workers, episodes 200 s, PPO epochs 4.

| Arm | Seed | Stage | Updates in stage | Warm-up | J sep (budget, lambda) | J corr (budget, lambda) | Fallback | KL | State |
|---|---|---|---|---|---|---|---|---|---|
| lag_orig2 | 1 | 20 | 15/312 | 0 | 0.336 (0.384, 0.41) | 0.48 (0.18, 1.14) | 0.000 | 0.0398 | training (attempt 1) |
| lag_orig2 | 2 | 20 | 18/312 | 0 | 0.523 (0.690, 0.43) | 0.02 (0.22, 1.09) | 0.000 | 0.0298 | training (attempt 1) |
| lag_orig2 | 3 | 20 | 19/312 | 0 | 0.720 (0.474, 1.26) | 1.39 (0.10, 1.75) | 0.000 | 0.0649 | training (attempt 1) |
| lag_orig2 | 4 | 20 | 22/312 | 0 | 0.716 (0.367, 2.35) | 0.31 (0.10, 2.56) | 0.000 | 0.0588 | training (attempt 1) |
| lag_teach | 1 | 15 | 0/312 | | | | | | abandoned |
| lag_teach | 2 | 15 | 0/312 | | | | | | abandoned |
| lag_teach | 3 | 15 | 0/312 | | | | | | abandoned |
| lag_teach | 4 | 15 | 0/312 | | | | | | abandoned |

Failures: {"lag_teach/3": "attempt 2 failed at 24 Nov 16:32 (see holiday_train_lag_teach_seed3.log)", "lag_teach/1": "attempt 2 failed at 24 Nov 16:32 (see holiday_train_lag_teach_seed1.log)", "lag_teach/4": "attempt 2 failed at 24 Nov 16:32 (see holiday_train_lag_teach_seed4.log)", "lag_teach/2": "attempt 2 failed at 24 Nov 16:32 (see holiday_train_lag_teach_seed2.log)"}


