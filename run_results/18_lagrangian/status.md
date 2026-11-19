# Constrained training - status (Thu 19 Nov 2026 20:15 server time)

Phase: **training**. Started Wed 18 Nov 20:15. Stages [10, 15, 20], 1000000 steps per stage (~312 updates), 16 workers, episodes 200 s, PPO epochs 4.

| Arm | Seed | Stage | Updates in stage | Warm-up | J sep (budget, lambda) | J corr (budget, lambda) | Fallback | KL | State |
|---|---|---|---|---|---|---|---|---|---|
| lag_final | 1 | 15 | 219/312 | 0 | 1.858 (2.632, 0.00) | 0.01 (0.19, 0.33) | 0.000 | 0.0522 | training (attempt 1) |
| lag_final | 2 | 15 | 217/312 | 0 | 3.288 (3.580, 0.00) | 0.08 (0.25, 0.00) | 0.000 | 0.0758 | training (attempt 1) |
| lag_final | 3 | 15 | 217/312 | 0 | 2.908 (2.692, 1.09) | 0.54 (0.08, 10.00) | 0.000 | 0.1294 | training (attempt 1) |
| lag_final | 4 | 15 | 217/312 | 0 | 2.712 (3.170, 0.20) | 0.45 (0.13, 10.00) | 0.001 | 0.0559 | training (attempt 1) |
| lag_orig | 1 | 15 | 236/312 | 0 | 0.931 (1.138, 2.44) | 0.11 (0.13, 6.16) | 0.000 | 0.0686 | training (attempt 1) |
| lag_orig | 2 | 15 | 234/312 | 0 | 1.167 (1.669, 0.00) | 0.04 (0.28, 0.00) | 0.000 | 0.0839 | training (attempt 1) |
| lag_orig | 3 | 15 | 242/312 | 0 | 0.692 (1.165, 0.00) | 0.12 (0.28, 0.65) | 0.000 | 0.0643 | training (attempt 1) |
| lag_orig | 4 | 15 | 262/312 | 0 | 0.419 (1.295, 5.03) | 0.02 (0.06, 10.00) | 0.000 | 0.0611 | training (attempt 1) |

Failures: {}


