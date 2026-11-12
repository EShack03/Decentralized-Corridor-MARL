"""[ES] added: two-way river corridor (held-out realistic layout, 12 Nov 2026).

Modelled on the Hudson River corridor in New York (14 CFR 93.352): one corridor along
a winding river carries both directions, each direction keeping to its own shore
(southbound along the west shore, northbound along the east shore). Here: two
one-way lanes, 300 m wide, centre lines 0.6 km apart (each 0.3 km to the right of
travel from the river centre line), each a chain of 4 straight segments following
the river through bends of about 23 deg, with 0.22 km gaps at the bends. Half of
the aircraft fly eastbound, half westbound (alternating), queued behind their own
lane's first entrance, 0.914 km apart.

Questions it answers: two-way traffic in one corridor (does the filter react to
aircraft in the opposite lane 600 m away?) and following through gentle bends.
Built on NetworkScenario (three_phase_graph_network_base); no original file is
modified. Specification fixed in advance (project doc, 12 Nov 2026).
"""
import numpy as np

from multiagent.custom_scenarios.three_phase_graph_network_base import NetworkScenario

RIVER = [(-3.2, -0.5), (-1.4, 0.05), (0.4, -0.15), (2.2, 0.4), (3.6, 0.25)]   # centre line, km
OFFSET = 0.3        # lane centre to the right of travel, km
GAP = 0.22          # gap at each bend, km
WIDTH = 0.3


def _lane(points, reverse=False):
	P = np.array(points, float)[::-1] if reverse else np.array(points, float)
	tubes = []
	for i in range(len(P) - 1):
		u = (P[i + 1] - P[i]) / np.linalg.norm(P[i + 1] - P[i])
		right = np.array([u[1], -u[0]])
		a = P[i] + OFFSET * right + GAP * u
		b = P[i + 1] + OFFSET * right - GAP * u
		tubes.append((tuple(a), tuple(b), WIDTH))
	return tubes


class Scenario(NetworkScenario):
	TUBES = _lane(RIVER) + _lane(RIVER, reverse=True)      # 0-3 eastbound, 4-7 westbound
	ROUTES = [[0, 1, 2, 3], [4, 5, 6, 7]]
	LATERAL = 0.15
