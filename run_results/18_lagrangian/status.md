# Constrained training - status (Fri 20 Nov 2026 14:15 server time)

Phase: **training**. Started Wed 18 Nov 20:15. Stages [10, 15, 20], 1000000 steps per stage (~312 updates), 16 workers, episodes 200 s, PPO epochs 4.

| Arm | Seed | Stage | Updates in stage | Warm-up | J sep (budget, lambda) | J corr (budget, lambda) | Fallback | KL | State |
|---|---|---|---|---|---|---|---|---|---|
| lag_final | 1 | 20 | 94/312 | 0 | 2.569 (2.189, 1.36) | 0.03 (0.01, 10.00) | 0.001 | 0.0888 | training (attempt 1) |
| lag_final | 2 | 20 | 90/312 | 0 | 2.373 (2.471, 0.00) | 0.12 (0.36, 0.00) | 0.000 | 0.0611 | training (attempt 1) |
| lag_final | 3 | 20 | 98/312 | 0 | 2.836 (2.509, 0.49) | 0.42 (0.01, 10.00) | 0.001 | 0.0542 | training (attempt 1) |
| lag_final | 4 | 20 | 92/312 | 0 | 2.803 (3.122, 0.00) | 0.00 (0.01, 10.00) | 0.000 | 0.0278 | training (attempt 1) |
| lag_orig | 1 | 20 | 126/312 | 0 | 0.352 (0.541, 3.39) | 0.01 (0.01, 10.00) | 0.000 | 0.0444 | training (attempt 1) |
| lag_orig | 2 | 20 | 121/312 | 0 | 0.552 (0.592, 5.23) | 0.64 (0.04, 10.00) | 0.000 | 0.1070 | training (attempt 1) |
| lag_orig | 3 | 20 | 121/312 | 0 | 0.477 (0.613, 1.36) | 0.01 (0.16, 3.72) | 0.000 | 0.0883 | training (attempt 1) |
| lag_orig | 4 | 20 | 146/312 | 0 | 0.555 (0.418, 0.34) | 0.03 (0.27, 0.00) | 0.000 | 0.0563 | training (attempt 1) |

Failures: {}


