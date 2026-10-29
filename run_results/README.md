# run_results/ — every evaluation result, filed by study

New runs write their result files to the repo root; `python organise_results.py --apply`
files them here (re-run it after each batch). Folder numbers follow the Run Results Ledger.

| Folder | What is in it | Ledger |
|---|---|---|
| `01_stage_pipeline/` | Stage 2 -> 3A -> 3B chain on the sequential corridor: original policy with no / static / constant-velocity filter (S2), horizon sweep of the original policy (HZ), 3A and 3B policies (S3A, S3B), 3 seeds x 50 episodes; plus the early density sweeps and reproduction checks | 1.0-1.7 |
| `02_horizon_sweep_3B/` | 3B policy, horizon 1-5, static vs constant velocity, 100 episodes (H3B) | 1.3 |
| `03_adaptive_rung1/` | Rung-1 adaptive alpha only (AD) | - |
| `04_fixed_vs_adaptive/<scenario>/` | fixed settings h3/a1, h1/a2, h1/a1.5 (HLfix, HLcomp) and the Rung-2 adaptive filter without the gap guard (HLadapt, ADH_h3) | 2.1-2.3, 2.6 |
| `05_stall_trigger/` | Rung-3 stall trigger and its threshold sweep (superseded) | 2.4 |
| `06_gap_guard/<scenario>/` | **final algorithm**: Rung-2 adaptive + junction gap guard (ADH_h3_gap) | 2.5, 2.7 |
| `07_no_filter/<scenario>/` | no filter: original policy (BASE) and 3B policy (S3B) | 2.5-2.7 |
| `08_other/` | any other recognised result | - |
| `09_smoke_tests/` | runs of 5 episodes or fewer | - |
| `_superseded/` | config-bug era, early smoke runs, superseded sweeps | provenance doc |
| `manifests/` | run lists written by `repro.py reproduce` (local only, not in git) | - |

**File names:** `<config>_<policy>_<filter>_<scenario>N<agents>_<episodes>ep_seed<k>_<YYMMDD>.txt`,
e.g. `ADH_h3_gap_shaped-train_cv_mergeN15_30ep_seed1_261025.txt` = final algorithm, 3B policy,
constant-velocity filter, merge, 15 aircraft, 30 episodes, seed 1, run on 25 Oct 2026.
Each `.txt` has a per-episode twin `<same name>_perep.csv` (kept locally; `.gitignore` excludes csv).
Scenarios: `seq` = sequential, `merge`, `single`, `three_phase_graph_merge_sequential` = mergeseq.
Note: the episode length is not in the name; see the Ledger for which runs used 200 s or 300 s.

**Collating:** `collate_repro.py` reads one folder, e.g.
`python collate_repro.py --dir run_results/06_gap_guard/mergeseq`.
