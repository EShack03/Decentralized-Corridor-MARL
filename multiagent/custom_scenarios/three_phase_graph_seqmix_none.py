"""[ES] added: sequential bend, all three off (= original sequential, control). See three_phase_graph_seqmix.py."""
from multiagent.custom_scenarios import three_phase_graph_seqmix as _m


class Scenario(_m.Scenario):
	VERT_GOAL = False
	VERT_SPAWN = False
	VERT_WIDTH = False
