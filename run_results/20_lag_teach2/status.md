# Constrained training - status (Wed 25 Nov 2026 04:54 server time)

Phase: **training**. Started Tue 24 Nov 16:54. Stages [10, 15, 20], 1000000 steps per stage (~312 updates), 16 workers, episodes 200 s, PPO epochs 4.

| Arm | Seed | Stage | Updates in stage | Warm-up | J sep (budget, lambda) | J corr (budget, lambda) | Fallback | KL | State |
|---|---|---|---|---|---|---|---|---|---|
| lag_teach2 | 1 | 15 | 44/312 | 0 | 0.726 (1.644, 0.00) | 1.79 (0.41, 3.00) | 0.000 | 0.0245 | training (attempt 1) |
| lag_teach2 | 2 | 15 | 39/312 | 0 | 1.239 (1.921, 0.00) | 0.61 (0.20, 3.00) | 0.000 | 0.0214 | training (attempt 1) |
| lag_teach2 | 3 | 15 | 41/312 | 0 | 1.174 (1.558, 0.00) | 2.30 (0.60, 3.00) | 0.000 | 0.0234 | training (attempt 1) |
| lag_teach2 | 4 | 15 | 36/312 | 0 | 0.548 (1.702, 0.00) | 2.01 (0.16, 3.00) | 0.000 | 0.0171 | training (attempt 1) |

Failures: {}


