# Constrained training - status (Fri 27 Nov 2026 04:54 server time)

Phase: **training**. Started Tue 24 Nov 16:54. Stages [10, 15, 20], 1000000 steps per stage (~312 updates), 16 workers, episodes 200 s, PPO epochs 4.

| Arm | Seed | Stage | Updates in stage | Warm-up | J sep (budget, lambda) | J corr (budget, lambda) | Fallback | KL | State |
|---|---|---|---|---|---|---|---|---|---|
| lag_teach2 | 1 | 20 | 297/312 | 0 | 0.777 (0.918, 0.00) | 0.16 (0.30, 2.93) | 0.000 | 0.0252 | training (attempt 1) |
| lag_teach2 | 2 | 20 | 287/312 | 0 | 0.964 (1.057, 0.00) | 0.17 (0.20, 3.00) | 0.000 | 0.0291 | training (attempt 1) |
| lag_teach2 | 3 | 20 | 283/312 | 0 | 0.561 (0.651, 0.09) | 0.04 (0.49, 0.00) | 0.000 | 0.0269 | training (attempt 1) |
| lag_teach2 | 4 | 20 | 274/312 | 0 | 0.614 (0.870, 0.00) | 0.08 (0.16, 3.00) | 0.000 | 0.0269 | training (attempt 1) |

Failures: {}


