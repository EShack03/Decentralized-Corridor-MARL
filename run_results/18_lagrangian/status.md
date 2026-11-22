# Constrained training - status (Sun 22 Nov 2026 02:15 server time)

Phase: **evaluation (588 runs)**. Started Wed 18 Nov 20:15. Stages [10, 15, 20], 1000000 steps per stage (~312 updates), 16 workers, episodes 200 s, PPO epochs 4.

| Arm | Seed | Stage | Updates in stage | Warm-up | J sep (budget, lambda) | J corr (budget, lambda) | Fallback | KL | State |
|---|---|---|---|---|---|---|---|---|---|
| lag_final | 1 | all done | 312/312 | 0 | 1.423 (2.189, 0.96) | 0.00 (0.01, 10.00) | 0.000 | 0.0938 | trained |
| lag_final | 2 | all done | 312/312 | 0 | 2.706 (2.471, 0.05) | 0.01 (0.36, 0.00) | 0.000 | 0.0814 | trained |
| lag_final | 3 | all done | 312/312 | 0 | 2.425 (2.509, 0.75) | 0.01 (0.01, 10.00) | 0.000 | 0.0899 | trained |
| lag_final | 4 | all done | 312/312 | 0 | 3.156 (3.122, 0.02) | 0.00 (0.01, 10.00) | 0.001 | 0.0725 | trained |
| lag_orig | 1 | all done | 312/312 | 0 | 0.534 (0.541, 2.29) | 0.03 (0.01, 10.00) | 0.000 | 0.0677 | trained |
| lag_orig | 2 | all done | 312/312 | 0 | 0.537 (0.592, 2.37) | 0.02 (0.04, 10.00) | 0.000 | 0.1061 | trained |
| lag_orig | 3 | all done | 312/312 | 0 | 0.636 (0.613, 0.21) | 0.00 (0.16, 0.17) | 0.000 | 0.0573 | trained |
| lag_orig | 4 | all done | 312/312 | 0 | 0.455 (0.418, 0.18) | 0.02 (0.27, 0.00) | 0.000 | 0.0899 | trained |

Failures: {}


