# Constrained training - status (Fri 20 Nov 2026 02:15 server time)

Phase: **training**. Started Wed 18 Nov 20:15. Stages [10, 15, 20], 1000000 steps per stage (~312 updates), 16 workers, episodes 200 s, PPO epochs 4.

| Arm | Seed | Stage | Updates in stage | Warm-up | J sep (budget, lambda) | J corr (budget, lambda) | Fallback | KL | State |
|---|---|---|---|---|---|---|---|---|---|
| lag_final | 1 | 15 | 303/312 | 0 | 2.962 (2.632, 0.16) | 0.01 (0.19, 0.93) | 0.000 | 0.0562 | training (attempt 1) |
| lag_final | 2 | 15 | 297/312 | 0 | 2.298 (3.580, 0.00) | 0.28 (0.25, 0.00) | 0.000 | 0.0848 | training (attempt 1) |
| lag_final | 3 | 15 | 305/312 | 0 | 2.844 (2.692, 0.57) | 0.00 (0.08, 5.62) | 0.000 | 0.1003 | training (attempt 1) |
| lag_final | 4 | 15 | 301/312 | 0 | 3.125 (3.170, 0.75) | 0.00 (0.13, 10.00) | 0.000 | 0.0519 | training (attempt 1) |
| lag_orig | 1 | 20 | 8/312 | 0 | 0.945 (0.541, 0.76) | 0.01 (0.01, 0.00) | 0.000 | 0.0122 | training (attempt 1) |
| lag_orig | 2 | 20 | 5/312 | 1 | 0.427 (0.592, 0.00) | 0.01 (0.04, 0.00) | 0.000 | 0.0000 | training (attempt 1) |
| lag_orig | 3 | 20 | 8/312 | 0 | 0.853 (0.613, 0.69) | 0.88 (0.16, 3.51) | 0.000 | 0.0111 | training (attempt 1) |
| lag_orig | 4 | 20 | 26/312 | 0 | 0.503 (0.418, 0.68) | 0.12 (0.27, 0.00) | 0.000 | 0.0477 | training (attempt 1) |

Failures: {}


