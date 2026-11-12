"""[ES] added: successive joins along a trunk (held-out realistic layout, 12 Nov 2026).

Modelled on London's helicopter routes, where feeder routes join the Thames route
at several points. A trunk runs west -> east in three straight segments (2.6 km to
the first junction, then to the second, then 2.0 km on; 0.167 km gaps at each
junction). Two feeders, 2.0 km long, join at 30 deg from alternating sides (the
first from the south-west, the second from the north-west), each ending 0.6 km
short of its junction so that it does not overlap the trunk. All corridors 300 m
wide. Demand 40 % trunk, 30 % per feeder: the trunk queue is 0.914 km apart and
the feeder queues 1.219 km apart, so the three flows arrive at rates 40:30:30.

Questions it answers: does joining get harder as the stream being joined gets
busier (40 % at the first junction, 70 % at the second)? Built on NetworkScenario;
no original file is modified. Specification fixed in advance (project doc, 12 Nov 2026).
"""
import numpy as np

from multiagent.custom_scenarios.three_phase_graph_network_base import NetworkScenario

J1, J2 = np.array([-1.0, 0.4]), np.array([1.6, 0.4])
G = 5.0 / 30           # trunk gap at each junction, km
FG = 0.6               # feeder stops this far short of its junction, km
E = np.array([1.0, 0.0])


def _dir(deg):
	return np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])


def _t(a, b, w=0.3):
	return (tuple(a), tuple(b), w)


class Scenario(NetworkScenario):
	TUBES = [
		_t(J1 - E * (2.6 + G), J1 - E * G),                    # 0: trunk to junction 1
		_t(J1 + E * G, J2 - E * G),                            # 1: trunk between junctions
		_t(J2 + E * G, J2 + E * (2.0 + G)),                    # 2: trunk after junction 2
		_t(J1 - _dir(30) * (2.0 + FG), J1 - _dir(30) * FG),    # 3: feeder 1 (from the south-west)
		_t(J2 - _dir(-30) * (2.0 + FG), J2 - _dir(-30) * FG),  # 4: feeder 2 (from the north-west)
	]
	ROUTES = [[0, 1, 2], [3, 1, 2], [4, 2]]
	ROUTE_CYCLE = [0, 1, 0, 2, 0, 1, 0, 2, 1, 2]          # 40 / 30 / 30
	LATERAL = 0.15
	ROUTE_SPACING = [0.914, 1.219, 1.219]                  # km, so flow rates are 40:30:30

	def spawn_offset(self, route, k):
		return (k + 1) * self.ROUTE_SPACING[route]
