# Constrained training - status (Tue 24 Nov 2026 22:54 server time)

Phase: **training**. Started Tue 24 Nov 16:54. Stages [10, 15, 20], 1000000 steps per stage (~312 updates), 16 workers, episodes 200 s, PPO epochs 4.

| Arm | Seed | Stage | Updates in stage | Warm-up | J sep (budget, lambda) | J corr (budget, lambda) | Fallback | KL | State |
|---|---|---|---|---|---|---|---|---|---|
| lag_teach2 | 1 | 10 | 223/312 | 0 | 2.310 (3.255, 0.00) | 0.00 (0.41, 3.00) | 0.000 | 0.0995 | training (attempt 1) |
| lag_teach2 | 2 | 10 | 221/312 | 0 | 2.323 (4.159, 0.00) | 0.17 (0.34, 3.00) | 0.000 | 0.0871 | training (attempt 1) |
| lag_teach2 | 3 | 10 | 221/312 | 0 | 3.671 (3.072, 0.07) | 1.89 (0.78, 3.00) | 0.000 | 0.0875 | training (attempt 1) |
| lag_teach2 | 4 | 10 | 214/312 | 0 | 3.538 (3.850, 0.00) | 0.31 (0.16, 3.00) | 0.000 | 0.0915 | training (attempt 1) |

Failures: {}


