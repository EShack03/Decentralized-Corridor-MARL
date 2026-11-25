# Constrained training - status (Wed 25 Nov 2026 10:54 server time)

Phase: **training**. Started Tue 24 Nov 16:54. Stages [10, 15, 20], 1000000 steps per stage (~312 updates), 16 workers, episodes 200 s, PPO epochs 4.

| Arm | Seed | Stage | Updates in stage | Warm-up | J sep (budget, lambda) | J corr (budget, lambda) | Fallback | KL | State |
|---|---|---|---|---|---|---|---|---|---|
| lag_teach2 | 1 | 15 | 149/312 | 0 | 1.119 (1.644, 0.00) | 2.03 (0.41, 3.00) | 0.000 | 0.0311 | training (attempt 1) |
| lag_teach2 | 2 | 15 | 141/312 | 0 | 2.163 (1.921, 0.03) | 0.40 (0.20, 3.00) | 0.000 | 0.0319 | training (attempt 1) |
| lag_teach2 | 3 | 15 | 142/312 | 0 | 0.826 (1.558, 0.00) | 1.57 (0.60, 3.00) | 0.000 | 0.0328 | training (attempt 1) |
| lag_teach2 | 4 | 15 | 135/312 | 0 | 1.197 (1.702, 0.00) | 0.37 (0.16, 3.00) | 0.000 | 0.0276 | training (attempt 1) |

Failures: {}


