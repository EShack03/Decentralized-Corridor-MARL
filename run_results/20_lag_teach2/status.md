# Constrained training - status (Wed 25 Nov 2026 22:54 server time)

Phase: **training**. Started Tue 24 Nov 16:54. Stages [10, 15, 20], 1000000 steps per stage (~312 updates), 16 workers, episodes 200 s, PPO epochs 4.

| Arm | Seed | Stage | Updates in stage | Warm-up | J sep (budget, lambda) | J corr (budget, lambda) | Fallback | KL | State |
|---|---|---|---|---|---|---|---|---|---|
| lag_teach2 | 1 | 20 | 12/312 | 0 | 0.533 (0.918, 0.00) | 1.65 (0.30, 1.62) | 0.000 | 0.0065 | training (attempt 1) |
| lag_teach2 | 2 | 20 | 3/312 | 1 | 1.147 (1.018, 0.00) | 0.52 (0.20, 0.00) | 0.000 | 0.0000 | training (attempt 1) |
| lag_teach2 | 3 | 20 | 7/312 | 0 | 0.839 (0.651, 0.33) | 0.62 (0.49, 0.24) | 0.000 | 0.0039 | training (attempt 1) |
| lag_teach2 | 4 | 15 | 309/312 | 0 | 1.034 (1.702, 0.00) | 0.20 (0.16, 3.00) | 0.000 | 0.0377 | training (attempt 1) |

Failures: {}


