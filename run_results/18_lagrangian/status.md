# Constrained training - status (Fri 20 Nov 2026 08:15 server time)

Phase: **training**. Started Wed 18 Nov 20:15. Stages [10, 15, 20], 1000000 steps per stage (~312 updates), 16 workers, episodes 200 s, PPO epochs 4.

| Arm | Seed | Stage | Updates in stage | Warm-up | J sep (budget, lambda) | J corr (budget, lambda) | Fallback | KL | State |
|---|---|---|---|---|---|---|---|---|---|
| lag_final | 1 | 20 | 43/312 | 0 | 3.383 (2.189, 1.10) | 0.31 (0.01, 10.00) | 0.000 | 0.0538 | training (attempt 1) |
| lag_final | 2 | 20 | 40/312 | 0 | 2.159 (2.471, 0.06) | 0.04 (0.36, 0.33) | 0.000 | 0.0312 | training (attempt 1) |
| lag_final | 3 | 20 | 46/312 | 0 | 2.908 (2.509, 0.20) | 0.01 (0.01, 10.00) | 0.001 | 0.0525 | training (attempt 1) |
| lag_final | 4 | 20 | 42/312 | 0 | 2.920 (3.122, 0.00) | 0.03 (0.01, 10.00) | 0.000 | 0.0350 | training (attempt 1) |
| lag_orig | 1 | 20 | 64/312 | 0 | 1.255 (0.541, 4.34) | 0.00 (0.01, 10.00) | 0.000 | 0.0329 | training (attempt 1) |
| lag_orig | 2 | 20 | 60/312 | 0 | 0.800 (0.592, 4.60) | 0.01 (0.04, 10.00) | 0.000 | 0.0895 | training (attempt 1) |
| lag_orig | 3 | 20 | 59/312 | 0 | 0.367 (0.613, 1.25) | 0.08 (0.16, 4.24) | 0.000 | 0.0690 | training (attempt 1) |
| lag_orig | 4 | 20 | 85/312 | 0 | 0.498 (0.418, 0.81) | 0.25 (0.27, 1.04) | 0.000 | 0.0330 | training (attempt 1) |

Failures: {}


