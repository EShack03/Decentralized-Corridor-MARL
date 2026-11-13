"""[ES] added: three_phase_graph_sequential_entrygoal with the goal-landmark nodes hidden from the
policy's graph (see goal_node_hiding.py)."""
from multiagent.custom_scenarios import three_phase_graph_sequential_entrygoal as _c
from multiagent.custom_scenarios.goal_node_hiding import HideGoalNodes


class Scenario(HideGoalNodes, _c.Scenario):
	pass
