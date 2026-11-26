# Constrained training - status (Thu 26 Nov 2026 16:54 server time)

Phase: **training**. Started Tue 24 Nov 16:54. Stages [10, 15, 20], 1000000 steps per stage (~312 updates), 16 workers, episodes 200 s, PPO epochs 4.

| Arm | Seed | Stage | Updates in stage | Warm-up | J sep (budget, lambda) | J corr (budget, lambda) | Fallback | KL | State |
|---|---|---|---|---|---|---|---|---|---|
| lag_teach2 | 1 | 20 | 182/312 | 0 | 0.745 (0.918, 0.00) | 0.52 (0.30, 3.00) | 0.000 | 0.0196 | training (attempt 1) |
| lag_teach2 | 2 | 20 | 173/312 | 0 | 1.197 (1.057, 0.06) | 0.43 (0.20, 3.00) | 0.000 | 0.0214 | training (attempt 1) |
| lag_teach2 | 3 | 20 | 172/312 | 0 | 0.922 (0.651, 0.40) | 0.13 (0.49, 1.80) | 0.000 | 0.0204 | training (attempt 1) |
| lag_teach2 | 4 | 20 | 164/312 | 0 | 0.723 (0.870, 0.00) | 0.17 (0.16, 3.00) | 0.000 | 0.0251 | training (attempt 1) |

Failures: {}


