# Constrained training - status (Sat 21 Nov 2026 02:15 server time)

Phase: **training**. Started Wed 18 Nov 20:15. Stages [10, 15, 20], 1000000 steps per stage (~312 updates), 16 workers, episodes 200 s, PPO epochs 4.

| Arm | Seed | Stage | Updates in stage | Warm-up | J sep (budget, lambda) | J corr (budget, lambda) | Fallback | KL | State |
|---|---|---|---|---|---|---|---|---|---|
| lag_final | 1 | 20 | 195/312 | 0 | 2.106 (2.189, 1.14) | 0.43 (0.01, 10.00) | 0.000 | 0.0842 | training (attempt 1) |
| lag_final | 2 | 20 | 187/312 | 0 | 2.952 (2.471, 0.18) | 0.27 (0.36, 0.00) | 0.000 | 0.0695 | training (attempt 1) |
| lag_final | 3 | 20 | 201/312 | 0 | 2.705 (2.509, 0.68) | 0.01 (0.01, 10.00) | 0.000 | 0.0551 | training (attempt 1) |
| lag_final | 4 | 20 | 190/312 | 0 | 2.678 (3.122, 0.00) | 0.00 (0.01, 10.00) | 0.000 | 0.0550 | training (attempt 1) |
| lag_orig | 1 | 20 | 246/312 | 0 | 0.845 (0.541, 2.93) | 0.05 (0.01, 10.00) | 0.000 | 0.0455 | training (attempt 1) |
| lag_orig | 2 | 20 | 241/312 | 0 | 0.366 (0.592, 3.28) | 0.20 (0.04, 10.00) | 0.000 | 0.1094 | training (attempt 1) |
| lag_orig | 3 | 20 | 241/312 | 0 | 0.452 (0.613, 0.50) | 0.00 (0.16, 3.12) | 0.000 | 0.0706 | training (attempt 1) |
| lag_orig | 4 | 20 | 263/312 | 0 | 0.447 (0.418, 0.23) | 0.05 (0.27, 0.00) | 0.000 | 0.0691 | training (attempt 1) |

Failures: {}


