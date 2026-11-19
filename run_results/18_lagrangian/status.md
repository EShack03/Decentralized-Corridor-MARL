# Constrained training - status (Thu 19 Nov 2026 08:15 server time)

Phase: **training**. Started Wed 18 Nov 20:15. Stages [10, 15, 20], 1000000 steps per stage (~312 updates), 16 workers, episodes 200 s, PPO epochs 4.

| Arm | Seed | Stage | Updates in stage | Warm-up | J sep (budget, lambda) | J corr (budget, lambda) | Fallback | KL | State |
|---|---|---|---|---|---|---|---|---|---|
| lag_final | 1 | 15 | 32/312 | 0 | 2.273 (2.632, 0.06) | 0.10 (0.19, 1.03) | 0.000 | 0.0256 | training (attempt 1) |
| lag_final | 2 | 15 | 42/312 | 0 | 3.104 (3.580, 0.00) | 0.72 (0.25, 1.75) | 0.000 | 0.0398 | training (attempt 1) |
| lag_final | 3 | 15 | 28/312 | 0 | 2.729 (2.692, 0.14) | 0.34 (0.08, 10.00) | 0.000 | 0.0582 | training (attempt 1) |
| lag_final | 4 | 15 | 31/312 | 0 | 3.300 (3.170, 0.03) | 0.00 (0.13, 4.46) | 0.000 | 0.0239 | training (attempt 1) |
| lag_orig | 1 | 15 | 43/312 | 0 | 1.052 (1.138, 2.56) | 0.04 (0.13, 3.18) | 0.000 | 0.0400 | training (attempt 1) |
| lag_orig | 2 | 15 | 42/312 | 0 | 2.033 (1.669, 2.41) | 0.37 (0.28, 0.90) | 0.000 | 0.0613 | training (attempt 1) |
| lag_orig | 3 | 15 | 52/312 | 0 | 1.602 (1.165, 2.48) | 0.05 (0.28, 1.67) | 0.000 | 0.0913 | training (attempt 1) |
| lag_orig | 4 | 15 | 46/312 | 0 | 2.250 (1.295, 5.49) | 0.13 (0.06, 10.00) | 0.000 | 0.0343 | training (attempt 1) |

Failures: {}


