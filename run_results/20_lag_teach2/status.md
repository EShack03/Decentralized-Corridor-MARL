# Constrained training - status (Thu 26 Nov 2026 04:54 server time)

Phase: **training**. Started Tue 24 Nov 16:54. Stages [10, 15, 20], 1000000 steps per stage (~312 updates), 16 workers, episodes 200 s, PPO epochs 4.

| Arm | Seed | Stage | Updates in stage | Warm-up | J sep (budget, lambda) | J corr (budget, lambda) | Fallback | KL | State |
|---|---|---|---|---|---|---|---|---|---|
| lag_teach2 | 1 | 20 | 67/312 | 0 | 0.681 (0.918, 0.00) | 0.47 (0.30, 3.00) | 0.000 | 0.0151 | training (attempt 1) |
| lag_teach2 | 2 | 20 | 58/312 | 0 | 1.028 (1.057, 0.00) | 0.29 (0.20, 3.00) | 0.000 | 0.0123 | training (attempt 1) |
| lag_teach2 | 3 | 20 | 61/312 | 0 | 0.758 (0.651, 0.08) | 0.42 (0.49, 0.82) | 0.000 | 0.0167 | training (attempt 1) |
| lag_teach2 | 4 | 20 | 53/312 | 0 | 0.739 (0.870, 0.00) | 0.08 (0.16, 3.00) | 0.000 | 0.0132 | training (attempt 1) |

Failures: {}


