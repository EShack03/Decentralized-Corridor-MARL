# Constrained training - status (Fri 20 Nov 2026 20:15 server time)

Phase: **training**. Started Wed 18 Nov 20:15. Stages [10, 15, 20], 1000000 steps per stage (~312 updates), 16 workers, episodes 200 s, PPO epochs 4.

| Arm | Seed | Stage | Updates in stage | Warm-up | J sep (budget, lambda) | J corr (budget, lambda) | Fallback | KL | State |
|---|---|---|---|---|---|---|---|---|---|
| lag_final | 1 | 20 | 143/312 | 0 | 1.830 (2.189, 1.04) | 0.01 (0.01, 10.00) | 0.000 | 0.1055 | training (attempt 1) |
| lag_final | 2 | 20 | 138/312 | 0 | 2.242 (2.471, 0.00) | 0.53 (0.36, 0.20) | 0.000 | 0.0611 | training (attempt 1) |
| lag_final | 3 | 20 | 149/312 | 0 | 2.072 (2.509, 0.39) | 0.01 (0.01, 10.00) | 0.000 | 0.0685 | training (attempt 1) |
| lag_final | 4 | 20 | 140/312 | 0 | 2.759 (3.122, 0.00) | 0.01 (0.01, 10.00) | 0.001 | 0.0447 | training (attempt 1) |
| lag_orig | 1 | 20 | 185/312 | 0 | 0.580 (0.541, 3.05) | 0.01 (0.01, 10.00) | 0.000 | 0.0389 | training (attempt 1) |
| lag_orig | 2 | 20 | 181/312 | 0 | 0.945 (0.592, 5.38) | 0.57 (0.04, 10.00) | 0.000 | 0.1204 | training (attempt 1) |
| lag_orig | 3 | 20 | 182/312 | 0 | 0.658 (0.613, 1.66) | 0.01 (0.16, 3.56) | 0.000 | 0.0830 | training (attempt 1) |
| lag_orig | 4 | 20 | 204/312 | 0 | 0.309 (0.418, 0.04) | 0.06 (0.27, 0.00) | 0.000 | 0.0488 | training (attempt 1) |

Failures: {}


