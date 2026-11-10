"""[ES] added: vertiport turnaround layout (scenario O).

Two arrival arms (from the north-west and the north-east) merge into a final-approach
corridor that leads down towards the vertiport. A short bottom corridor runs west ->
east past the vertiport, and a climb-out corridor runs back up, parallel to final
approach but flown the other way, before a departure arm turns away to the
north-east. Every aircraft flies the whole loop:

    arrival arm -> final approach -> bottom corridor -> climb-out -> departure arm

so the turnaround is made through two 90-degree junctions, each laid out like the
sequential layout's bend (next corridor's entrance 0.83 km down and 0.83 km across
from the previous exit, the geometry the policy already flies there). The parallel
corridors are 2.4 km apart (0.83 + 0.74 km bottom corridor + 0.83). Hand-offs use
the original rule (on leaving a corridor through its exit gate, the goal moves to
just past the next corridor's exit). The simulation has no time on the ground, so
this is a turnaround past the pad, not a landing. Aircraft start queued behind the
two arrival arms in turn, 0.91 km apart (the original merge's queue spacing).

Design history (tests of 10 Nov 2026, 10 aircraft, 1 episode): with an open turn and
no bottom corridor, climb-out 0.8 km away, aircraft could not make the U-turn (turn
rate limit 0.075 rad/s = 1.2 km radius at 90 m/s) and looped below the pad; at 2.4 km
they made it but some joined the climb-out from the side and missed its entrance
gate. The departure arm starts 0.85 km beyond and 0.35 km right of the climb-out
exit and turns only 45 degrees so the turn onto it fits.

Questions it answers: a merge, a guided turnaround through two tight junctions, and
opposite-direction traffic in parallel corridors, all in one airspace. Built on the
merge scenario through three_phase_graph_network_base.NetworkScenario; no original
file is modified.
"""
from multiagent.custom_scenarios.three_phase_graph_network_base import NetworkScenario

_D = 5.0 / 6     # junction offset used by the sequential layout (world_size / 6)


class Scenario(NetworkScenario):
	TUBES = [
		((-2.4, 2.1), (-0.75, 0.85), 0.3),               # 0: arrival arm 1 (from the north-west)
		((0.8, 2.3), (-0.2, 0.95), 0.3),                 # 1: arrival arm 2 (from the north-east)
		((-0.45, 0.55), (-0.45, -1.7), 0.4),             # 2: final approach (down)
		((1.95, -1.7), (1.95, 0.6), 0.4),                # 3: climb-out (up, 2.4 km to the right)
		((2.3, 1.45), (3.6, 2.75), 0.3),                 # 4: departure arm (45 deg right)
		((-0.45 + _D, -1.7 - _D), (1.95 - _D, -1.7 - _D), 0.4),  # 5: bottom corridor past the vertiport (west -> east)
	]
	ROUTES = [[0, 2, 5, 3, 4], [1, 2, 5, 3, 4]]
