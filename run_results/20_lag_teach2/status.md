# Constrained training - status (Thu 26 Nov 2026 22:54 server time)

Phase: **training**. Started Tue 24 Nov 16:54. Stages [10, 15, 20], 1000000 steps per stage (~312 updates), 16 workers, episodes 200 s, PPO epochs 4.

| Arm | Seed | Stage | Updates in stage | Warm-up | J sep (budget, lambda) | J corr (budget, lambda) | Fallback | KL | State |
|---|---|---|---|---|---|---|---|---|---|
| lag_teach2 | 1 | 20 | 239/312 | 0 | 0.833 (0.918, 0.00) | 0.90 (0.30, 3.00) | 0.000 | 0.0242 | training (attempt 1) |
| lag_teach2 | 2 | 20 | 229/312 | 0 | 0.983 (1.057, 0.00) | 0.45 (0.20, 3.00) | 0.000 | 0.0288 | training (attempt 1) |
| lag_teach2 | 3 | 20 | 226/312 | 0 | 0.812 (0.651, 0.15) | 0.62 (0.49, 1.37) | 0.000 | 0.0245 | training (attempt 1) |
| lag_teach2 | 4 | 20 | 218/312 | 0 | 0.722 (0.870, 0.00) | 0.42 (0.16, 3.00) | 0.000 | 0.0246 | training (attempt 1) |

Failures: {}


