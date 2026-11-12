"""[ES] added: original sequential layout with the goal-landmark nodes hidden from the
policy's graph (see goal_node_hiding.py). Everything else is the original scenario."""
from multiagent.custom_scenarios import three_phase_graph_sequential as _seq
from multiagent.custom_scenarios.goal_node_hiding import hide_goal_nodes


class Scenario(_seq.Scenario):
	def graph_observation(self, agent, world):
		node_obs, adj = super().graph_observation(agent, world)
		return node_obs, hide_goal_nodes(adj, world)
