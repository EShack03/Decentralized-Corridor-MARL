"""[ES] added: branching trunk (held-out realistic layout, 12 Nov 2026).

One trunk (400 m wide, 3.3 km) splits towards two destinations: two branches, 300 m
wide and 3.2 km long, at +25 and -25 deg, starting 0.4 km past the split point so
they do not overlap (the trunk ends 0.167 km before it). Destinations alternate
50/50 and are mixed in a single queue behind the trunk, 0.4 km apart (about 2.6x
the 152 m separation), so some aircraft must move across a dense stream to reach
their branch. (The first specification used 0.6 km; one-episode checks with no
filter showed no losses of separation at 0.6 or 0.5 km, i.e. no conflicts to test,
while 0.4 km was the loosest spacing that produced conflicts; the queue was
tightened to 0.4 km before any test runs. Disclosed in the project doc.)

Questions it answers: diverging traffic (sideways movement inside a stream), a
different conflict from merging. Built on NetworkScenario; no original file is
modified. Specification fixed in advance (project doc, 12 Nov 2026).
"""
import numpy as np

from multiagent.custom_scenarios.three_phase_graph_network_base import NetworkScenario

D = np.array([-0.4, 0.3])      # split point, km
G_IN, G_OUT = 5.0 / 30, 0.4


def _dir(deg):
	return np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])


class Scenario(NetworkScenario):
	TUBES = [
		(tuple(D - _dir(0) * (3.3 + G_IN)), tuple(D - _dir(0) * G_IN), 0.4),     # 0: trunk
		(tuple(D + _dir(25) * G_OUT), tuple(D + _dir(25) * (3.2 + G_OUT)), 0.3),  # 1: branch north
		(tuple(D + _dir(-25) * G_OUT), tuple(D + _dir(-25) * (3.2 + G_OUT)), 0.3),  # 2: branch south
	]
	ROUTES = [[0, 1], [0, 2]]
	LATERAL = 0.15
	QUEUE_SPACING = 0.4           # km, one shared queue (~2.6x separation)

	def spawn_offset(self, route, k):
		# routes alternate (round robin), so aircraft i = k * 2 + route stands i-th in the queue
		return (k * len(self.ROUTES) + route + 1) * self.QUEUE_SPACING
