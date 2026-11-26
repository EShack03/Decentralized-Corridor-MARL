# Constrained training - status (Thu 26 Nov 2026 10:54 server time)

Phase: **training**. Started Tue 24 Nov 16:54. Stages [10, 15, 20], 1000000 steps per stage (~312 updates), 16 workers, episodes 200 s, PPO epochs 4.

| Arm | Seed | Stage | Updates in stage | Warm-up | J sep (budget, lambda) | J corr (budget, lambda) | Fallback | KL | State |
|---|---|---|---|---|---|---|---|---|---|
| lag_teach2 | 1 | 20 | 124/312 | 0 | 0.806 (0.918, 0.00) | 0.52 (0.30, 3.00) | 0.000 | 0.0161 | training (attempt 1) |
| lag_teach2 | 2 | 20 | 115/312 | 0 | 0.950 (1.057, 0.00) | 0.37 (0.20, 3.00) | 0.000 | 0.0180 | training (attempt 1) |
| lag_teach2 | 3 | 20 | 116/312 | 0 | 0.747 (0.651, 0.06) | 0.08 (0.49, 1.26) | 0.000 | 0.0187 | training (attempt 1) |
| lag_teach2 | 4 | 20 | 108/312 | 0 | 0.977 (0.870, 0.04) | 0.72 (0.16, 3.00) | 0.000 | 0.0192 | training (attempt 1) |

Failures: {}


