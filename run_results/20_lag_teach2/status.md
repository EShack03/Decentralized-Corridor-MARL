# Constrained training - status (Wed 25 Nov 2026 16:54 server time)

Phase: **training**. Started Tue 24 Nov 16:54. Stages [10, 15, 20], 1000000 steps per stage (~312 updates), 16 workers, episodes 200 s, PPO epochs 4.

| Arm | Seed | Stage | Updates in stage | Warm-up | J sep (budget, lambda) | J corr (budget, lambda) | Fallback | KL | State |
|---|---|---|---|---|---|---|---|---|---|
| lag_teach2 | 1 | 15 | 241/312 | 0 | 1.233 (1.644, 0.00) | 0.82 (0.41, 3.00) | 0.000 | 0.0323 | training (attempt 1) |
| lag_teach2 | 2 | 15 | 227/312 | 0 | 0.903 (1.921, 0.00) | 0.11 (0.20, 2.18) | 0.000 | 0.0344 | training (attempt 1) |
| lag_teach2 | 3 | 15 | 232/312 | 0 | 0.904 (1.558, 0.00) | 1.57 (0.60, 3.00) | 0.000 | 0.0366 | training (attempt 1) |
| lag_teach2 | 4 | 15 | 220/312 | 0 | 0.972 (1.702, 0.00) | 0.88 (0.16, 3.00) | 0.000 | 0.0355 | training (attempt 1) |

Failures: {}


