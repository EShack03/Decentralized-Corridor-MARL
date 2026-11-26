# Constrained training - status (Thu 26 Nov 2026 08:40 server time)

Phase: **training**. Started Tue 24 Nov 02:40. Stages [10, 15, 20], 1000000 steps per stage (~312 updates), 16 workers, episodes 200 s, PPO epochs 4.

| Arm | Seed | Stage | Updates in stage | Warm-up | J sep (budget, lambda) | J corr (budget, lambda) | Fallback | KL | State |
|---|---|---|---|---|---|---|---|---|---|
| lag_orig2 | 1 | 20 | 242/312 | 0 | 0.125 (0.384, 0.00) | 0.10 (0.18, 0.00) | 0.000 | 0.0636 | training (attempt 1) |
| lag_orig2 | 2 | 20 | 230/312 | 0 | 0.484 (0.690, 0.00) | 0.03 (0.22, 0.00) | 0.000 | 0.0626 | training (attempt 1) |
| lag_orig2 | 3 | 20 | 239/312 | 0 | 0.773 (0.474, 0.47) | 0.03 (0.10, 0.00) | 0.000 | 0.0955 | training (attempt 1) |
| lag_orig2 | 4 | 20 | 263/312 | 0 | 0.373 (0.367, 0.58) | 0.03 (0.10, 1.63) | 0.000 | 0.0795 | training (attempt 1) |
| lag_teach | 1 | 15 | 0/312 | | | | | | abandoned |
| lag_teach | 2 | 15 | 0/312 | | | | | | abandoned |
| lag_teach | 3 | 15 | 0/312 | | | | | | abandoned |
| lag_teach | 4 | 15 | 0/312 | | | | | | abandoned |

Failures: {"lag_teach/3": "attempt 2 failed at 24 Nov 16:32 (see holiday_train_lag_teach_seed3.log)", "lag_teach/1": "attempt 2 failed at 24 Nov 16:32 (see holiday_train_lag_teach_seed1.log)", "lag_teach/4": "attempt 2 failed at 24 Nov 16:32 (see holiday_train_lag_teach_seed4.log)", "lag_teach/2": "attempt 2 failed at 24 Nov 16:32 (see holiday_train_lag_teach_seed2.log)"}


