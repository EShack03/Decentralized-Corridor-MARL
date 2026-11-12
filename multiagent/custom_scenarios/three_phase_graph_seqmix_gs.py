"""[ES] added: sequential bend with the vertiport's goal spacing, spawn. See three_phase_graph_seqmix.py."""
from multiagent.custom_scenarios import three_phase_graph_seqmix as _m


class Scenario(_m.Scenario):
	VERT_GOAL = True
	VERT_SPAWN = True
	VERT_WIDTH = False
