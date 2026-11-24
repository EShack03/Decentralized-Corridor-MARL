# Constrained training - status (Tue 24 Nov 2026 08:40 server time)

Phase: **training**. Started Tue 24 Nov 02:40. Stages [10, 15, 20], 1000000 steps per stage (~312 updates), 16 workers, episodes 200 s, PPO epochs 4.

| Arm | Seed | Stage | Updates in stage | Warm-up | J sep (budget, lambda) | J corr (budget, lambda) | Fallback | KL | State |
|---|---|---|---|---|---|---|---|---|---|
| lag_orig2 | 1 | 10 | 242/312 | 0 | 4.516 (3.888, 0.04) | 0.46 (0.32, 1.02) | 0.000 | 0.1414 | training (attempt 1) |
| lag_orig2 | 2 | 10 | 246/312 | 0 | 2.372 (5.369, 0.00) | 0.10 (0.43, 0.00) | 0.000 | 0.1091 | training (attempt 1) |
| lag_orig2 | 3 | 10 | 249/312 | 0 | 1.788 (3.919, 0.00) | 0.79 (0.79, 0.00) | 0.000 | 0.1232 | training (attempt 1) |
| lag_orig2 | 4 | 10 | 257/312 | 0 | 3.853 (4.927, 0.00) | 0.02 (0.15, 1.04) | 0.000 | 0.0980 | training (attempt 1) |
| lag_teach | 1 | 10 | 250/312 | 0 | 0.042 (3.255, 0.00) | 29.40 (0.41, 3.00) | 0.000 | 0.7162 | training (attempt 1) |
| lag_teach | 2 | 10 | 249/312 | 0 | 0.036 (4.159, 0.00) | 30.39 (0.34, 3.00) | 0.000 | 0.5488 | training (attempt 1) |
| lag_teach | 3 | 10 | 246/312 | 0 | 0.233 (3.072, 0.00) | 20.83 (0.78, 3.00) | 0.000 | 0.5511 | training (attempt 1) |
| lag_teach | 4 | 10 | 258/312 | 0 | 0.000 (3.850, 0.00) | 34.66 (0.16, 3.00) | 0.000 | 0.7605 | training (attempt 1) |

Failures: {}


