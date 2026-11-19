# Constrained training - status (Thu 19 Nov 2026 02:15 server time)

Phase: **training**. Started Wed 18 Nov 20:15. Stages [10, 15, 20], 1000000 steps per stage (~312 updates), 16 workers, episodes 200 s, PPO epochs 4.

| Arm | Seed | Stage | Updates in stage | Warm-up | J sep (budget, lambda) | J corr (budget, lambda) | Fallback | KL | State |
|---|---|---|---|---|---|---|---|---|---|
| lag_final | 1 | 10 | 163/312 | 0 | 3.094 (3.134, 1.15) | 0.11 (0.28, 10.00) | 0.000 | 0.1131 | training (attempt 1) |
| lag_final | 2 | 10 | 190/312 | 0 | 3.991 (4.071, 0.25) | 0.02 (0.24, 10.00) | 0.000 | 0.1141 | training (attempt 1) |
| lag_final | 3 | 10 | 161/312 | 0 | 3.291 (3.084, 0.41) | 0.06 (0.54, 2.47) | 0.000 | 0.1295 | training (attempt 1) |
| lag_final | 4 | 10 | 165/312 | 0 | 3.475 (3.687, 0.59) | 0.72 (0.11, 10.00) | 0.000 | 0.1323 | training (attempt 1) |
| lag_orig | 1 | 10 | 184/312 | 0 | 1.762 (3.888, 5.12) | 0.16 (0.22, 10.00) | 0.000 | 0.1758 | training (attempt 1) |
| lag_orig | 2 | 10 | 178/312 | 0 | 3.600 (5.369, 3.68) | 0.07 (0.30, 10.00) | 0.000 | 0.1899 | training (attempt 1) |
| lag_orig | 3 | 10 | 205/312 | 0 | 2.972 (3.919, 1.32) | 0.16 (0.56, 4.27) | 0.000 | 0.1494 | training (attempt 1) |
| lag_orig | 4 | 10 | 186/312 | 0 | 3.916 (4.927, 5.79) | 0.81 (0.11, 10.00) | 0.000 | 0.2094 | training (attempt 1) |

Failures: {}


