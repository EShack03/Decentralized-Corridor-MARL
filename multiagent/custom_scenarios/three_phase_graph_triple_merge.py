"""[ES] added: triple merge onto a final approach (held-out realistic layout, 12 Nov 2026).

Vertiport arrivals: three inbound routes, 2.6 km long and 300 m wide, travelling at
60, 0 and -60 deg to the final-approach axis, converge on one merge point (each arm
ends 0.35 km short of it so the arms do not overlap) and continue down one final
approach (400 m wide, 2.3 km, starting 0.167 km past the merge point). One third of
the aircraft per arm, queued 0.914 km apart behind each arm.

Questions it answers: three streams sequenced onto one bottleneck, no merge steeper
than 60 deg. Built on NetworkScenario; no original file is modified. Specification
fixed in advance (project doc, 12 Nov 2026).
"""
import numpy as np

from multiagent.custom_scenarios.three_phase_graph_network_base import NetworkScenario

M = np.array([0.0, 0.9])        # merge point, km
ARM, ARM_GAP = 2.6, 0.35
FIN, FIN_GAP = 2.3, 5.0 / 30


def _arm(deg):
	d = np.array([np.sin(np.radians(deg)), -np.cos(np.radians(deg))])   # direction of travel
	return (tuple(M - d * (ARM + ARM_GAP)), tuple(M - d * ARM_GAP), 0.3)


class Scenario(NetworkScenario):
	TUBES = [_arm(60), _arm(0), _arm(-60),
			 ((0.0, M[1] - FIN_GAP), (0.0, M[1] - FIN_GAP - FIN), 0.4)]
	ROUTES = [[0, 3], [1, 3], [2, 3]]
	LATERAL = 0.15
