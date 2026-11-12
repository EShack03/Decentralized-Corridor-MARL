"""[ES] added (12 Nov 2026): hide goal-landmark nodes from the policy's graph.

The single-corridor (working_three_phase_graph) and sequential scenario files create one
Landmark entity per aircraft, all at the current corridor's goal, and keep every one of them
in the graph for the whole episode (the "reached goal" mask keys on goal_tracker, which is
never updated). The merge family (merge, merge-sequential and every layout built on them)
has no landmark entities at all: its authors removed them because "they bloat the graph".

hide_goal_nodes() removes every edge to and from the landmark nodes in the adjacency matrix
the policy receives, so the actor (which reads the ego node after message passing) is no
longer influenced by them. Node features, rewards, goals, hand-offs, success and every other
part of the scenario are untouched; each aircraft still sees its own goal in its own
observation and node features. This changes what the original policy sees, so it is a
flagged experiment.
"""


def hide_goal_nodes(adj, world):
	if not len(world.landmarks):
		return adj
	adj = adj.copy()                          # never write into world.cached_dist_mag
	lm = slice(len(world.agents), len(world.agents) + len(world.landmarks))   # entity order: agents, landmarks, obstacles, walls
	adj[lm, :] = 0
	adj[:, lm] = 0
	return adj
