"""[ES] added: crossing layout (scenario A).

Two independent one-corridor routes cross at 90 degrees: west -> east and
north -> south. Nothing merges; each route is a single corridor like the original
single-corridor layout (4 km long, 600 m wide; the policy was trained on that
layout), so the only new element is crossing traffic: aircraft must time the gaps
in the other flow at the intersection, where both corridors overlap.

Aircraft alternate between the two routes and queue behind their own entrance
0.91 km apart (the original merge's queue spacing). Goals sit past each corridor's
exit (merge convention, world_size / 12). Built on the merge scenario through
three_phase_graph_network_base.NetworkScenario; no original file is modified.
"""
from multiagent.custom_scenarios.three_phase_graph_network_base import NetworkScenario


class Scenario(NetworkScenario):
	TUBES = [
		((-2.0, 0.0), (2.0, 0.0), 0.6),     # 0: west -> east
		((0.0, 2.0), (0.0, -2.0), 0.6),     # 1: north -> south
	]
	ROUTES = [[0], [1]]
