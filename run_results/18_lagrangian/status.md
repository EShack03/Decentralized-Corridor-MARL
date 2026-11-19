# Constrained training - status (Thu 19 Nov 2026 14:15 server time)

Phase: **training**. Started Wed 18 Nov 20:15. Stages [10, 15, 20], 1000000 steps per stage (~312 updates), 16 workers, episodes 200 s, PPO epochs 4.

| Arm | Seed | Stage | Updates in stage | Warm-up | J sep (budget, lambda) | J corr (budget, lambda) | Fallback | KL | State |
|---|---|---|---|---|---|---|---|---|---|
| lag_final | 1 | 15 | 130/312 | 0 | 2.265 (2.632, 0.00) | 0.02 (0.19, 0.18) | 0.000 | 0.0337 | training (attempt 1) |
| lag_final | 2 | 15 | 132/312 | 0 | 2.485 (3.580, 0.00) | 0.01 (0.25, 0.00) | 0.000 | 0.0636 | training (attempt 1) |
| lag_final | 3 | 15 | 124/312 | 0 | 2.115 (2.692, 1.17) | 0.01 (0.08, 9.59) | 0.000 | 0.0874 | training (attempt 1) |
| lag_final | 4 | 15 | 126/312 | 0 | 2.835 (3.170, 0.16) | 0.00 (0.13, 10.00) | 0.000 | 0.0495 | training (attempt 1) |
| lag_orig | 1 | 15 | 138/312 | 0 | 1.283 (1.138, 4.24) | 0.03 (0.13, 8.57) | 0.000 | 0.0475 | training (attempt 1) |
| lag_orig | 2 | 15 | 144/312 | 0 | 1.296 (1.669, 1.50) | 0.02 (0.28, 1.98) | 0.000 | 0.0602 | training (attempt 1) |
| lag_orig | 3 | 15 | 150/312 | 0 | 1.056 (1.165, 0.00) | 0.35 (0.28, 0.01) | 0.000 | 0.0697 | training (attempt 1) |
| lag_orig | 4 | 15 | 157/312 | 0 | 1.131 (1.295, 9.14) | 0.01 (0.06, 10.00) | 0.000 | 0.0628 | training (attempt 1) |

Failures: {}


