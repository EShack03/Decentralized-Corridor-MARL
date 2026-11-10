# New corridor layouts (added 10 Nov 2026)

All new layouts are **new files** in `multiagent/custom_scenarios/`; no original scenario
file is changed. They reuse the original scenarios' dynamics, observations, graph,
reward terms, phase/gate rules and success rule (an aircraft succeeds when it exits its
final corridor correctly). A figure of the real corridor geometry with one episode of
paths per layout is kept outside the repo (PNG files are git-ignored).

| File (`--scenario_name`) | Results folder | What it is | Question it answers |
|---|---|---|---|
| `three_phase_graph_sequential_entrygoal` | `seq_entrygoal` | Sequential bend; while in corridor 1 the observed target is corridor 2's entrance instead of 1.25 km straight past corridor 1's exit | Does the far-away intermediate target cause the conflicts at the bend? |
| `three_phase_graph_crossing` | `crossing` | Two one-corridor routes (4 km x 600 m, like the single corridor) crossing at 90 deg | Crossing traffic with no merge |
| `three_phase_graph_merge_angle90` / `_angle180` | `merge90` / `merge180` | The merge with its arms 90 / 180 deg apart; `_angle90` reproduces `three_phase_graph_merge` exactly (checked: identical metrics, seed 1); 180 deg is a T-junction (arm exits 330 m apart, facing each other) | Head-on approach to a merge (stress case; real procedures join at about 30 deg, at most 45) |
| `three_phase_graph_merge_unbalanced` | `unbalanced` | The merge with 80 % of aircraft on one arm, arriving in bunches of 3 (0.46 km apart, 3 km between bunches) | Platoons at the merge; is the light arm starved? |
| `three_phase_graph_vertiport` | `vertiport` | Two arrival arms merge onto final approach down towards the vertiport; every aircraft turns round through a short bottom corridor past the pad (two 90 deg junctions laid out like the sequential bend), climbs back up a parallel corridor flown the other way (2.4 km away) and leaves by a departure arm | A merge, a guided turnaround and opposite-direction traffic in one airspace |

`three_phase_graph_network_base.py` holds the shared code (not runnable by itself):
the merge geometry with a chosen arm angle, and `NetworkScenario`, which builds any
layout from a list of corridors and routes (any number of corridors per route) and
reuses the merge reward unchanged by showing it the aircraft's current and next
corridor for the duration of that one call.

Evaluation queue: `queue_newscen.txt` (no filter vs the final filter: constant
velocity, ignore parked, 3 s, no learned barrier, separation x1.2; 10/15/20 aircraft;
3 seeds x 30 episodes; 300 s). Tags `NS_none_tanh` / `NS_F12_tanh` go to
`run_results/16_new_scenarios/<folder>/`. `_angle90` is not in the queue because it
is identical to the existing merge results.
