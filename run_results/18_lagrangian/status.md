# Constrained training - status (Sat 21 Nov 2026 08:15 server time)

Phase: **training**. Started Wed 18 Nov 20:15. Stages [10, 15, 20], 1000000 steps per stage (~312 updates), 16 workers, episodes 200 s, PPO epochs 4.

| Arm | Seed | Stage | Updates in stage | Warm-up | J sep (budget, lambda) | J corr (budget, lambda) | Fallback | KL | State |
|---|---|---|---|---|---|---|---|---|---|
| lag_final | 1 | 20 | 251/312 | 0 | 2.336 (2.189, 1.10) | 0.05 (0.01, 10.00) | 0.000 | 0.1182 | training (attempt 1) |
| lag_final | 2 | 20 | 239/312 | 0 | 2.309 (2.471, 0.09) | 0.09 (0.36, 0.00) | 0.000 | 0.0614 | training (attempt 1) |
| lag_final | 3 | 20 | 255/312 | 0 | 2.773 (2.509, 0.92) | 0.31 (0.01, 10.00) | 0.001 | 0.0716 | training (attempt 1) |
| lag_final | 4 | 20 | 244/312 | 0 | 2.953 (3.122, 0.00) | 0.00 (0.01, 10.00) | 0.001 | 0.0518 | training (attempt 1) |
| lag_orig | 1 | 20 | 309/312 | 0 | 0.506 (0.541, 2.20) | 0.00 (0.01, 10.00) | 0.000 | 0.0621 | training (attempt 1) |
| lag_orig | 2 | 20 | 305/312 | 0 | 0.597 (0.592, 2.30) | 0.00 (0.04, 10.00) | 0.000 | 0.0999 | training (attempt 1) |
| lag_orig | 3 | 20 | 300/312 | 0 | 0.216 (0.613, 0.00) | 0.02 (0.16, 0.70) | 0.000 | 0.0492 | training (attempt 1) |
| lag_orig | 4 | all done | 312/312 | 0 | 0.455 (0.418, 0.18) | 0.02 (0.27, 0.00) | 0.000 | 0.0899 | trained |

Failures: {}


