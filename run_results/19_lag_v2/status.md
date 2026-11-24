# Constrained training - status (Tue 24 Nov 2026 14:40 server time)

Phase: **training**. Started Tue 24 Nov 02:40. Stages [10, 15, 20], 1000000 steps per stage (~312 updates), 16 workers, episodes 200 s, PPO epochs 4.

| Arm | Seed | Stage | Updates in stage | Warm-up | J sep (budget, lambda) | J corr (budget, lambda) | Fallback | KL | State |
|---|---|---|---|---|---|---|---|---|---|
| lag_orig2 | 1 | 15 | 66/312 | 0 | 0.856 (1.033, 2.52) | 0.35 (0.36, 2.21) | 0.000 | 0.0877 | training (attempt 1) |
| lag_orig2 | 2 | 15 | 67/312 | 0 | 1.385 (1.171, 3.33) | 0.56 (0.29, 3.00) | 0.000 | 0.1044 | training (attempt 1) |
| lag_orig2 | 3 | 15 | 70/312 | 0 | 1.231 (1.244, 1.76) | 1.00 (0.87, 1.36) | 0.000 | 0.1186 | training (attempt 1) |
| lag_orig2 | 4 | 15 | 76/312 | 0 | 0.894 (1.117, 2.53) | 0.09 (0.12, 3.00) | 0.000 | 0.1084 | training (attempt 1) |
| lag_teach | 1 | 15 | 85/312 | 0 | 0.003 (0.030, 1.45) | 17.13 (24.50, 0.88) | 0.000 | 0.1107 | training (attempt 1) |
| lag_teach | 2 | 15 | 81/312 | 0 | 0.011 (0.055, 0.65) | 23.99 (19.53, 3.00) | 0.000 | 0.1924 | training (attempt 1) |
| lag_teach | 3 | 15 | 90/312 | 0 | 0.012 (0.023, 4.64) | 28.90 (22.45, 2.23) | 0.000 | 0.1529 | training (attempt 1) |
| lag_teach | 4 | 15 | 91/312 | 0 | 0.037 (0.053, 0.41) | 26.30 (26.84, 1.26) | 0.000 | 0.1636 | training (attempt 1) |

Failures: {}


