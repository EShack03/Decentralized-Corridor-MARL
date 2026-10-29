# Reproduction report

Parsed **76** result files -> **76** configs (after seed-averaging) from `C:\MEng_Project_Code\Decentralized-Corridor-MARL`.

Format `old→new` = documented pre-repro 3-seed mean → this run (seed-averaged). Collisions = losses of separation per episode.

## Reconciliation (new vs documented)

| config | metric | old | new | Δ | verdict |
|---|---|---|---|---|---|
| 3A/cv N=5 | collisions | 0.06 | 0.02 | -0.04 | PASS |
| 3A/cv N=5 | success% | 64.70 | 65.60 | +0.90 | PASS |
| 3A/none N=5 | collisions | 53.79 | 57.35 | +3.56 | PASS |
| 3A/none N=5 | success% | 68.80 | 66.80 | -2.00 | PASS |
| 3B/cv N=5 | collisions | 0.05 | 0.06 | +0.00 | PASS |
| 3B/cv N=5 | success% | 94.70 | 94.20 | -0.50 | PASS |
| 3B/cv N=8 | collisions | 0.55 | 0.45 | -0.11 | PASS |
| 3B/cv N=8 | success% | 92.70 | 93.38 | +0.68 | PASS |
| 3B/cv N=10 | collisions | 1.33 | 1.19 | -0.15 | PASS |
| 3B/cv N=10 | success% | 92.70 | 93.20 | +0.50 | PASS |
| 3B/none N=5 | collisions | 6.13 | 9.71 | +3.58 | CHECK |
| 3B/none N=5 | success% | 95.30 | 94.60 | -0.70 | PASS |
| 3B/none N=8 | collisions | 22.54 | 27.56 | +5.02 | PASS |
| 3B/none N=8 | success% | 93.50 | 93.62 | +0.12 | PASS |
| 3B/none N=10 | collisions | 43.39 | 43.44 | +0.05 | PASS |
| 3B/none N=10 | success% | 94.10 | 94.40 | +0.30 | PASS |
| 3B/static N=5 | collisions | 0.87 | 1.16 | +0.29 | PASS |
| 3B/static N=5 | success% | 94.00 | 91.20 | -2.80 | PASS |
| 3B/static N=8 | collisions | 6.38 | 5.58 | -0.80 | PASS |
| 3B/static N=8 | success% | 91.90 | 92.88 | +0.98 | PASS |
| 3B/static N=10 | collisions | 14.33 | 11.63 | -2.70 | PASS |
| 3B/static N=10 | success% | 91.50 | 92.90 | +1.40 | PASS |
| original/cv N=5 | collisions | 0.05 | 0.10 | +0.05 | PASS |
| original/cv N=5 | success% | 89.20 | 88.40 | -0.80 | PASS |
| original/cv N=8 | collisions | 0.50 | 0.40 | -0.10 | PASS |
| original/cv N=8 | success% | 88.20 | 87.75 | -0.45 | PASS |
| original/cv N=10 | collisions | 1.11 | 1.31 | +0.20 | PASS |
| original/cv N=10 | success% | 86.70 | 85.20 | -1.50 | PASS |
| original/none N=5 | collisions | 11.72 | 11.01 | -0.71 | PASS |
| original/none N=5 | success% | 89.50 | 90.60 | +1.10 | PASS |
| original/none N=8 | collisions | 27.57 | 30.57 | +3.00 | PASS |
| original/none N=8 | success% | 90.20 | 88.50 | -1.70 | PASS |
| original/none N=10 | collisions | 32.85 | 34.13 | +1.28 | PASS |
| original/none N=10 | success% | 87.60 | 88.50 | +0.90 | PASS |
| original/static N=5 | collisions | 2.47 | 2.13 | -0.34 | PASS |
| original/static N=5 | success% | 82.90 | 83.00 | +0.10 | PASS |
| original/static N=8 | collisions | 12.61 | 13.92 | +1.31 | PASS |
| original/static N=8 | success% | 78.40 | 77.38 | -1.02 | PASS |
| original/static N=10 | collisions | 21.56 | 19.90 | -1.66 | PASS |
| original/static N=10 | success% | 76.10 | 76.00 | -0.10 | PASS |

**39 PASS, 1 CHECK** (band: collisions ±max(0.6, 25%); success ±6 pts).

## Stage 2 - original policy

### Original policy x filter

| N | none old→new coll | static old→new coll | cv old→new coll | none old→new succ% | static old→new succ% | cv old→new succ% |
|---|---|---|---|---|---|---|
| 5 | 11.72→11.01 | 2.47→2.13 | 0.05→0.10 | 89.5→90.6 | 82.9→83.0 | 89.2→88.4 |
| 8 | 27.57→30.57 | 12.61→13.92 | 0.50→0.40 | 90.2→88.5 | 78.4→77.4 | 88.2→87.8 |
| 10 | 32.85→34.13 | 21.56→19.90 | 1.11→1.31 | 87.6→88.5 | 76.1→76.0 | 86.7→85.2 |

## Stage 3A - filter-dependence diagnostic (N=5)

### 3A policy

| N | none old→new coll | cv old→new coll | none old→new succ% | cv old→new succ% |
|---|---|---|---|---|
| 5 | 53.79→57.35 | 0.06→0.02 | 68.8→66.8 | 64.7→65.6 |

## Stage 3B - shaped policy

### 3B policy x filter

| N | none old→new coll | static old→new coll | cv old→new coll | none old→new succ% | static old→new succ% | cv old→new succ% |
|---|---|---|---|---|---|---|
| 5 | 6.13→9.71 | 0.87→1.16 | 0.05→0.06 | 95.3→94.6 | 94.0→91.2 | 94.7→94.2 |
| 8 | 22.54→27.56 | 6.38→5.58 | 0.55→0.45 | 93.5→93.6 | 91.9→92.9 | 92.7→93.4 |
| 10 | 43.39→43.44 | 14.33→11.63 | 1.33→1.19 | 94.1→94.4 | 91.5→92.9 | 92.7→93.2 |

## Horizon

### Horizon sweep (new data - the result the sweep establishes)

collisions / no-safe-action% / success% by horizon and neighbour model. For the original policy h=3 is the Stage-2 run; longer CV horizons are the new HZ/H3B runs. Read along a row: static rises with horizon (compounding prediction error), CV falls then plateaus (useful look-ahead).

#### Original policy

| neighbour | N | h=1 | h=2 | h=3 | h=4 | h=5 |
|---|---|---|---|---|---|---|
| static | 5 | 1.66 / 26.1 / 85.2 | 1.74 / 25.0 / 85.2 | 2.13 / 27.9 / 83.0 | - | - |
| static | 8 | 6.38 / 28.8 / 86.8 | 8.59 / 27.8 / 83.9 | 13.92 / 26.9 / 77.4 | - | - |
| static | 10 | 10.11 / 27.1 / 82.3 | 14.87 / 23.9 / 77.7 | 19.90 / 24.5 / 76.0 | - | - |
| cv | 5 | 0.35 / 24.6 / 90.8 | 0.14 / 25.9 / 89.4 | 0.10 / 26.4 / 88.4 | 0.03 / 28.1 / 89.2 | 0.01 / 26.8 / 88.2 |
| cv | 8 | 2.25 / 30.3 / 88.6 | 1.13 / 28.7 / 90.2 | 0.40 / 28.8 / 87.8 | 0.38 / 31.1 / 88.9 | 0.25 / 29.3 / 87.5 |
| cv | 10 | 4.01 / 27.9 / 85.0 | 1.67 / 25.7 / 84.2 | 1.31 / 27.9 / 85.2 | 1.06 / 28.5 / 84.2 | 0.71 / 27.9 / 84.8 |

#### 3B (deployed) policy

| neighbour | N | h=1 | h=2 | h=3 | h=4 | h=5 |
|---|---|---|---|---|---|---|
| static | 5 | 0.73 / 19.9 / 95.6 | 0.98 / 22.4 / 92.4 | 1.16 / 25.2 / 91.2 | - | - |
| static | 8 | 3.26 / 26.2 / 93.6 | 4.12 / 27.2 / 93.6 | 5.58 / 27.4 / 92.9 | - | - |
| static | 10 | 9.31 / 27.4 / 93.6 | 11.21 / 28.4 / 91.6 | 11.63 / 29.0 / 92.9 | - | - |
| cv | 5 | 0.05 / 22.4 / 94.6 | 0.13 / 22.7 / 94.6 | 0.06 / 21.4 / 94.2 | 0.02 / 22.2 / 94.6 | 0.03 / 22.6 / 94.2 |
| cv | 8 | 1.26 / 27.8 / 91.9 | 0.49 / 24.7 / 94.2 | 0.45 / 27.8 / 93.4 | 0.35 / 27.4 / 94.2 | 0.33 / 29.1 / 93.9 |
| cv | 10 | 3.37 / 28.6 / 93.8 | 1.69 / 28.5 / 94.1 | 1.19 / 28.5 / 93.2 | 1.60 / 29.1 / 92.1 | 1.09 / 28.6 / 93.0 |

## Determinism

### Determinism / non-perturbation cross-check

_(no duplicate configs in this matrix: each config is produced under a single tag, so there is nothing to cross-check here. Bit-exact determinism / non-perturbation was established separately in the 3-seed reproduction, where the regenerated h=3 runs matched the Stage-2 runs to every decimal.)_

## Pooled statistics

### Pooled per-episode statistics (all runs of a config concatenated)

Each row concatenates every per-episode CSV for that config. SE = std/sqrt(n). p50/p95/max are the per-episode collision percentiles (the safety tail); coll-free = episodes with zero loss of separation.

| config | n_ep | mean coll | SE | p50 | p95 | max | coll-free % | mean succ% |
|---|---|---|---|---|---|---|---|---|
| merge:3B/cv N=10 h=1 [a1.0] | 100 | 3.965 | 0.418 | 3.00 | 12.00 | 18.00 | 28.0 | 91.4 |
| merge:3B/cv N=10 h=1 [a1.5] | 100 | 1.120 | 0.247 | 0.00 | 6.00 | 12.00 | 72.0 | 95.0 |
| merge:3B/cv N=10 h=1 [a2.0] | 100 | 0.615 | 0.145 | 0.00 | 3.00 | 9.00 | 78.0 | 96.0 |
| merge:3B/cv N=10 h=2 [a1.0] | 100 | 2.540 | 0.269 | 2.00 | 7.00 | 13.00 | 31.0 | 80.5 |
| merge:3B/cv N=10 h=2 [a1.5] | 100 | 1.965 | 0.247 | 1.00 | 7.00 | 12.00 | 39.0 | 81.7 |
| merge:3B/cv N=10 h=2 [a2.0] | 100 | 2.035 | 0.293 | 1.00 | 8.00 | 18.00 | 46.0 | 85.9 |
| merge:3B/cv N=10 h=3 | 100 | 1.895 | 0.266 | 1.00 | 6.00 | 16.00 | 45.0 | 59.6 |
| merge:3B/cv N=10 h=3 [a1.0] | 100 | 1.895 | 0.266 | 1.00 | 6.00 | 16.00 | 45.0 | 59.6 |
| merge:3B/cv N=10 h=3 [a1.5] | 100 | 1.495 | 0.205 | 1.00 | 6.00 | 9.00 | 48.0 | 65.6 |
| merge:3B/cv N=10 h=3 [a2.0] | 100 | 1.795 | 0.257 | 1.00 | 6.00 | 12.00 | 50.0 | 67.6 |
| merge:3B/cv N=5 h=3 | 100 | 0.140 | 0.047 | 0.00 | 1.00 | 3.00 | 90.0 | 58.8 |
| merge:3B/cv N=8 h=3 | 100 | 1.085 | 0.183 | 0.00 | 5.00 | 10.00 | 60.0 | 60.0 |
| merge:original/cv N=10 | 100 | 0.310 | 0.092 | 0.00 | 2.00 | 6.00 | 84.0 | 61.1 |
| merge:original/cv N=5 | 100 | 0.110 | 0.051 | 0.00 | 1.00 | 4.00 | 94.0 | 59.4 |
| merge:original/cv N=8 | 100 | 0.165 | 0.067 | 0.00 | 1.00 | 5.00 | 92.0 | 61.6 |
| merge:original/none N=10 | 100 | 4.025 | 0.886 | 0.00 | 24.00 | 25.50 | 82.0 | 99.4 |
| merge:original/none N=5 | 100 | 5.405 | 0.990 | 0.00 | 25.00 | 26.00 | 75.0 | 99.6 |
| merge:original/none N=8 | 100 | 4.055 | 0.865 | 0.00 | 24.50 | 25.00 | 78.0 | 99.4 |
| seq:3A/cv N=5 | 100 | 0.020 | 0.020 | 0.00 | 0.00 | 2.00 | 99.0 | 65.6 |
| seq:3A/none N=5 | 100 | 57.350 | 5.713 | 45.50 | 155.50 | 283.00 | 16.0 | 66.8 |
| seq:3B/cv N=10 h=1 | 100 | 3.365 | 0.439 | 2.00 | 12.50 | 23.00 | 27.0 | 93.8 |
| seq:3B/cv N=10 h=1 [a1.5] | 100 | 2.810 | 0.324 | 2.00 | 10.00 | 15.50 | 25.0 | 93.2 |
| seq:3B/cv N=10 h=1 [a2.0] | 100 | 3.200 | 0.409 | 2.00 | 10.50 | 27.00 | 31.0 | 91.1 |
| seq:3B/cv N=10 h=2 | 100 | 1.690 | 0.198 | 1.00 | 5.00 | 8.00 | 40.0 | 94.1 |
| seq:3B/cv N=10 h=3 | 100 | 1.185 | 0.153 | 1.00 | 5.00 | 5.00 | 46.0 | 93.2 |
| seq:3B/cv N=10 h=4 | 100 | 1.605 | 0.203 | 1.00 | 5.50 | 9.00 | 45.0 | 92.1 |
| seq:3B/cv N=10 h=5 | 100 | 1.095 | 0.135 | 1.00 | 4.00 | 6.00 | 46.0 | 93.0 |
| seq:3B/cv N=10 [A0.5] | 100 | 1.230 | 0.150 | 1.00 | 4.00 | 5.00 | 47.0 | 93.6 |
| seq:3B/cv N=10 [A0.75] | 100 | 1.350 | 0.197 | 1.00 | 5.00 | 8.00 | 48.0 | 92.6 |
| seq:3B/cv N=10 [FB] | 100 | 24.110 | 2.709 | 13.00 | 80.50 | 104.00 | 18.0 | 94.4 |
| seq:3B/cv N=5 h=1 | 100 | 0.050 | 0.026 | 0.00 | 0.00 | 2.00 | 96.0 | 94.6 |
| seq:3B/cv N=5 h=2 | 100 | 0.130 | 0.084 | 0.00 | 0.00 | 8.00 | 96.0 | 94.6 |
| seq:3B/cv N=5 h=3 | 100 | 0.055 | 0.026 | 0.00 | 0.00 | 2.00 | 96.0 | 94.2 |
| seq:3B/cv N=5 h=4 | 100 | 0.020 | 0.014 | 0.00 | 0.00 | 1.00 | 98.0 | 94.6 |
| seq:3B/cv N=5 h=5 | 100 | 0.030 | 0.022 | 0.00 | 0.00 | 2.00 | 98.0 | 94.2 |
| seq:3B/cv N=5 [FB] | 100 | 6.685 | 1.237 | 0.00 | 34.50 | 51.00 | 64.0 | 94.6 |
| seq:3B/cv N=8 h=1 | 100 | 1.265 | 0.195 | 0.00 | 5.00 | 9.00 | 55.0 | 91.9 |
| seq:3B/cv N=8 h=2 | 100 | 0.490 | 0.113 | 0.00 | 3.00 | 6.00 | 79.0 | 94.2 |
| seq:3B/cv N=8 h=3 | 100 | 0.445 | 0.091 | 0.00 | 2.00 | 4.00 | 76.0 | 93.4 |
| seq:3B/cv N=8 h=4 | 100 | 0.355 | 0.068 | 0.00 | 2.00 | 3.00 | 75.0 | 94.2 |
| seq:3B/cv N=8 h=5 | 100 | 0.330 | 0.107 | 0.00 | 2.00 | 9.00 | 82.0 | 93.9 |
| seq:3B/cv N=8 [FB] | 100 | 14.540 | 2.347 | 0.50 | 74.00 | 85.00 | 51.0 | 93.6 |
| seq:3B/none N=10 | 100 | 43.435 | 3.510 | 39.00 | 108.00 | 154.00 | 9.0 | 94.4 |
| seq:3B/none N=5 | 100 | 9.715 | 1.664 | 0.00 | 51.00 | 67.00 | 64.0 | 94.6 |
| seq:3B/none N=8 | 100 | 27.560 | 3.372 | 19.00 | 93.00 | 149.00 | 29.0 | 93.6 |
| seq:3B/static N=10 h=1 | 100 | 9.315 | 0.985 | 6.00 | 30.50 | 48.00 | 15.0 | 93.6 |
| seq:3B/static N=10 h=2 | 100 | 11.210 | 1.188 | 7.00 | 38.50 | 63.00 | 9.0 | 91.6 |
| seq:3B/static N=10 h=3 | 100 | 11.635 | 1.039 | 10.00 | 29.50 | 59.50 | 8.0 | 92.9 |
| seq:3B/static N=5 h=1 | 100 | 0.730 | 0.230 | 0.00 | 5.00 | 15.00 | 83.0 | 95.6 |
| seq:3B/static N=5 h=2 | 100 | 0.980 | 0.263 | 0.00 | 5.00 | 19.00 | 78.0 | 92.4 |
| seq:3B/static N=5 h=3 | 100 | 1.155 | 0.364 | 0.00 | 7.00 | 23.00 | 83.0 | 91.2 |
| seq:3B/static N=8 h=1 | 100 | 3.260 | 0.741 | 0.00 | 14.00 | 51.50 | 53.0 | 93.6 |
| seq:3B/static N=8 h=2 | 100 | 4.120 | 0.580 | 2.00 | 17.00 | 23.50 | 45.0 | 93.6 |
| seq:3B/static N=8 h=3 | 100 | 5.575 | 0.718 | 3.00 | 16.00 | 38.00 | 39.0 | 92.9 |
| seq:original/cv N=10 h=1 | 100 | 4.010 | 0.320 | 3.00 | 10.00 | 13.00 | 14.0 | 85.0 |
| seq:original/cv N=10 h=2 | 100 | 1.670 | 0.175 | 1.00 | 5.00 | 7.50 | 35.0 | 84.2 |
| seq:original/cv N=10 h=4 | 100 | 1.065 | 0.153 | 0.00 | 4.00 | 6.00 | 58.0 | 84.2 |
| seq:original/cv N=10 h=5 | 100 | 0.715 | 0.120 | 0.00 | 3.00 | 6.50 | 63.0 | 84.8 |
| seq:original/cv N=10 | 100 | 1.315 | 0.163 | 1.00 | 4.00 | 9.50 | 44.0 | 85.2 |
| seq:original/cv N=5 h=1 | 100 | 0.355 | 0.118 | 0.00 | 2.00 | 10.00 | 84.0 | 90.8 |
| seq:original/cv N=5 h=2 | 100 | 0.140 | 0.060 | 0.00 | 1.00 | 4.00 | 94.0 | 89.4 |
| seq:original/cv N=5 h=4 | 100 | 0.030 | 0.022 | 0.00 | 0.00 | 2.00 | 98.0 | 89.2 |
| seq:original/cv N=5 h=5 | 100 | 0.010 | 0.010 | 0.00 | 0.00 | 1.00 | 99.0 | 88.2 |
| seq:original/cv N=5 | 100 | 0.105 | 0.043 | 0.00 | 1.00 | 2.00 | 94.0 | 88.4 |
| seq:original/cv N=8 h=1 | 100 | 2.250 | 0.241 | 2.00 | 8.00 | 9.00 | 34.0 | 88.6 |
| seq:original/cv N=8 h=2 | 100 | 1.130 | 0.174 | 0.00 | 4.00 | 8.00 | 56.0 | 90.2 |
| seq:original/cv N=8 h=4 | 100 | 0.375 | 0.089 | 0.00 | 2.00 | 6.00 | 78.0 | 88.9 |
| seq:original/cv N=8 h=5 | 100 | 0.250 | 0.057 | 0.00 | 2.00 | 2.00 | 82.0 | 87.5 |
| seq:original/cv N=8 | 100 | 0.400 | 0.077 | 0.00 | 2.00 | 4.00 | 73.0 | 87.8 |
| seq:original/none N=10 | 100 | 34.130 | 2.796 | 30.00 | 89.00 | 121.00 | 6.0 | 88.5 |
| seq:original/none N=5 | 100 | 11.010 | 1.972 | 1.00 | 57.00 | 89.00 | 49.0 | 90.6 |
| seq:original/none N=8 | 100 | 30.575 | 2.750 | 27.00 | 84.50 | 103.00 | 11.0 | 88.5 |
| seq:original/static N=10 h=1 | 100 | 10.110 | 0.719 | 9.00 | 23.00 | 30.00 | 9.0 | 82.3 |
| seq:original/static N=10 h=2 | 100 | 14.865 | 1.389 | 12.00 | 42.50 | 88.00 | 5.0 | 77.7 |
| seq:original/static N=10 | 100 | 19.900 | 1.479 | 17.00 | 55.00 | 70.00 | 1.0 | 76.0 |
| seq:original/static N=5 h=1 | 100 | 1.655 | 0.368 | 0.00 | 4.50 | 27.00 | 63.0 | 85.2 |
| seq:original/static N=5 h=2 | 100 | 1.740 | 0.463 | 0.00 | 8.00 | 39.00 | 70.0 | 85.2 |
| seq:original/static N=5 | 100 | 2.130 | 0.397 | 0.00 | 10.00 | 25.00 | 64.0 | 83.0 |
| seq:original/static N=8 h=1 | 100 | 6.380 | 0.758 | 5.00 | 15.00 | 62.00 | 16.0 | 86.8 |
| seq:original/static N=8 h=2 | 100 | 8.590 | 1.315 | 6.00 | 26.00 | 97.00 | 18.0 | 83.9 |
| seq:original/static N=8 | 100 | 13.920 | 1.551 | 8.00 | 48.00 | 71.00 | 12.0 | 77.4 |
| single:3B/cv N=10 h=1 [a1.5] | 100 | 0.210 | 0.072 | 0.00 | 2.00 | 5.00 | 90.0 | 99.9 |
| single:3B/cv N=10 h=1 [a2.0] | 100 | 0.300 | 0.108 | 0.00 | 2.00 | 8.00 | 90.0 | 99.9 |
| single:3B/cv N=10 h=3 | 100 | 0.435 | 0.088 | 0.00 | 2.00 | 5.00 | 74.0 | 100.0 |
| single:3B/cv N=5 h=3 | 100 | 0.000 | 0.000 | 0.00 | 0.00 | 0.00 | 100.0 | 100.0 |
| single:3B/cv N=8 h=3 | 100 | 0.120 | 0.045 | 0.00 | 1.00 | 3.00 | 92.0 | 99.9 |
| single:original/none N=10 | 100 | 3.970 | 0.553 | 2.00 | 11.00 | 37.00 | 21.0 | 83.9 |
| single:original/none N=5 | 100 | 0.830 | 0.334 | 0.00 | 4.00 | 21.00 | 90.0 | 93.4 |
| single:original/none N=8 | 100 | 2.405 | 0.434 | 0.00 | 13.00 | 21.00 | 59.0 | 89.6 |

## Cross-scenario difficulty

### Cross-scenario difficulty (raw per-episode collisions)

Mean per-episode collisions (loss of separation). Compare DOWN a column (same policy/filter/N) across scenarios: higher = more unsafe. Raw counts are comparable only at equal episode length -- run single/merge at --episode_length 200 to match sequential.

| policy/filter N | single | merge | seq |
|---|---|---|---|
| 3A/cv N=5 | -- | -- | 0.02 |
| 3A/none N=5 | -- | -- | 57.35 |
| 3B/cv N=5 | 0.00 | 0.14 | 0.06 |
| 3B/cv N=8 | 0.12 | 1.08 | 0.45 |
| 3B/cv N=10 | 0.43 | 1.90 | 1.19 |
| 3B/none N=5 | -- | -- | 9.71 |
| 3B/none N=8 | -- | -- | 27.56 |
| 3B/none N=10 | -- | -- | 43.44 |
| 3B/static N=5 | -- | -- | 1.16 |
| 3B/static N=8 | -- | -- | 5.58 |
| 3B/static N=10 | -- | -- | 11.63 |
| original/cv N=5 | -- | 0.11 | 0.10 |
| original/cv N=8 | -- | 0.17 | 0.40 |
| original/cv N=10 | -- | 0.31 | 1.31 |
| original/none N=5 | 0.83 | 5.41 | 11.01 |
| original/none N=8 | 2.40 | 4.05 | 30.57 |
| original/none N=10 | 3.97 | 4.03 | 34.13 |
| original/static N=5 | -- | -- | 2.13 |
| original/static N=8 | -- | -- | 13.92 |
| original/static N=10 | -- | -- | 19.90 |

