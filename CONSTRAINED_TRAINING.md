# Constrained fine-tuning (Nov 2026)

Fine-tunes the original corridor policy so it is safer and keeps to its corridor,
starting from `model_weights/`. Everything is new files or flag-gated additions;
with no `--lag_*` flags, `train_mpe.py` trains exactly as before.

## Set-ups

| `--setup` | How the costs enter training | Filter in training |
|---|---|---|
| `penalty` | Standard MAPPO: fixed penalty `reward -= 10 x separation cost + 1 x corridor cost` | on |
| `lagrangian` (main) | MAPPO-Lagrangian: one cost critic per cost, PID multipliers hold each cost at its budget | on |
| `lag_nofilter` | MAPPO-Lagrangian, as above | off |

All three share the same warm-up, KL anchor, mask-consistency term, fallback
handling and training seeds, so they differ only in the column that changes.

## Costs (per aircraft, per second)

- **Separation:** 1 for each other active aircraft closer than 152 m, plus 0.25
  for each one 152-304 m away (`--lag_near_weight`). Parked aircraft are ignored.
- **Corridor:** 1 while the aircraft is off its corridor: it left the side of its
  current corridor between the entrance and the exit (lateral offset > half width
  + 50 m) and has not come back in. The start queue, the junction gap and the run
  to the goal after a proper exit never count.

The budget is the target for the mean cost per aircraft per episode. By default
it is measured from the starting policy during warm-up: separation budget = 1.0 x
the starting level (no worse than now), corridor budget = 0.7 x (30 % less time off-corridor).
In `lag_nofilter` the warm-up still runs with the filter, so the budget is the
level the original policy reaches *with* the filter, and the policy has to reach
it on its own.

## Pieces

| File | What it does |
|---|---|
| `onpolicy/envs/cost_mask_wrapper.py` | Runs in each rollout worker: adds costs and the plain-filter masks (constant-velocity neighbours, fixed look-ahead, alpha 1, parked aircraft ignored) to the infos |
| `onpolicy/algorithms/graph_mappo_lag.py` | `GR_MAPPO_Lag(GR_MAPPO)`: cost critics, cost GAE, PID multipliers, combined advantage, KL anchor, mask-consistency term; reuses `ppo_update` unchanged |
| `onpolicy/runner/shared/graph_mpe_runner_lag.py` | `GMPERunnerLag(GMPERunner)`: stores costs, applies the worker masks through the existing filter hook, saves `lag_state.pt` |
| `onpolicy/scripts/train_mpe.py` | `--lag_*` flags, layout mix across workers (`--lag_scenarios`), picks the runner |
| `train_lag.py` | Runs one set-up and seed through the 10 -> 15 -> 20 aircraft stages |

Two fixes to how the original runner uses the buffer, applied only in
`GMPERunnerLag`:

1. **Mask alignment.** `GraphReplayBuffer.insert` stores `available_actions` at
   `step + 1`, but the runner passes the mask the current action was sampled
   under. Left as is, the PPO ratio at step t uses step t-1's mask. The runner
   now also stores it at `step`.
2. **First step.** `buffer.after_update()` is never called (it is commented out
   in `base_runner.py`), so step 0 of every rollout acts on the observation from
   the very first reset. The runner now calls it after each update.

## Mixing layouts

Each rollout worker runs one layout (`--lag_scenarios`, worker k gets layout
k mod 4). The merge layouts' graphs hold the aircraft only, single and sequential
also hold the goal nodes, so every graph is padded to 2 x aircraft nodes
(`--lag_pad_nodes`, automatic). Padding nodes are all-zero with no edges, so the
actor's output is unchanged; only the critics' graph mean sees them.

## Restoring the critic

`critic.pt` has no PopArt running statistics (`v_out.mean`, `mean_sq`,
`debiasing_term`): PopArt creates them with `nn.Parameter(...).to(device)`, which
on a GPU returns an unregistered tensor, so the original run never saved them.
They restart at zero; the warm-up updates re-fit the critic before the actor
moves. Any other missing or unexpected key stops the run.

## Activation

The original policy was trained with Tanh hidden activations (`use_ReLU: false`
in `model_weights/config.yaml`). `train_lag.py` passes `--use_ReLU` (store_false,
so this means Tanh), and the critic restore is checked key by key. Evaluate the
fine-tuned policies with `--use_ReLU` as well.

## Commands

```
python train_lag.py --setup lagrangian --seed 1 --smoke      # ~minutes on CPU: checks it runs
python train_lag.py --setup lagrangian --seed 1 --dry_run    # print the stage commands
python train_lag.py --setup lagrangian --seed 1              # full run (GPU)
```

Each stage writes `lag_log.csv` (one row per update: costs, budgets, multipliers,
KL, fallback rate) next to its `models/` folder; a rerun skips finished stages.
