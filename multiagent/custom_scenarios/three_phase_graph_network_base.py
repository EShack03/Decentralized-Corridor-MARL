"""[ES] added: shared helpers for the new corridor layouts (not a runnable scenario by
itself; the runnable layouts are the small files that import this one).

Everything is built on the existing merge scenario (three_phase_graph_merge.py),
which is NOT modified: same air-taxi dynamics, observation, graph, reward terms,
phase/gate rules and success rule (an aircraft succeeds when it exits its final
corridor correctly). Only the corridor geometry, the route each aircraft follows and
where aircraft start are new.

1. set_merge_tubes(scen, world, included_deg)
   The merge geometry with the angle between the two arms as a parameter.
   included_deg = 90 reproduces three_phase_graph_merge.py exactly (its arms are
   45 deg either side of the stem axis).

2. NetworkScenario
   A layout given as a list of corridors (TUBES) and routes (ROUTES). Each route is a
   list of corridor indices flown in order ([k] = a one-corridor route). The merge
   reward hands an aircraft from corridor 0 or 1 to corridor 2 (moving its goal to
   just past corridor 2's exit) and treats corridor 2 as final. To reuse that reward
   unchanged for any route, reward() shows the merge code the aircraft's current
   corridor as index 0 and the next one on its route as index 2 (or the current one
   as index 2 if it is the last) for the duration of that one call, then restores
   the real layout and maps the aircraft back to its real corridor index. Nothing
   outside reward() ever sees the temporary order: observations, metrics, the safety
   filter and the corridor-membership checks all use the real layout.
   HANDOFF_AT_GOAL: corridor indices after which the hand-off to the next corridor
   happens only when the aircraft reaches its goal (within the scenario's
   goal-reached distance) instead of on leaving the corridor through its exit gate.
"""
import numpy as np

from multiagent.custom_scenarios import three_phase_graph_merge as _merge


def _tube(scen, entrance, exit_, width):
	"""Corridor dict in the merge scenario's format from end points (km)."""
	entrance = np.asarray(entrance, dtype=float)
	exit_ = np.asarray(exit_, dtype=float)
	length = float(np.linalg.norm(exit_ - entrance))
	e = (exit_ - entrance) / length
	# merge convention: rotation_matrix @ [0, -1] is the flow direction
	angle = float(np.arctan2(-e[0], -e[1]))
	R = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
	return scen._build_tube_params(entrance, exit_, width, angle, length, R)


def set_merge_tubes(scen, world, included_deg):
	"""Merge geometry (two arms -> merge point -> stem) with a chosen arm angle.
	Same widths, lengths and gap as three_phase_graph_merge.py."""
	world.tube_params = []
	tube_width = 0.3
	merge_point = np.array([0.0, 0.0])
	branch_length = scen.world_size * 0.3
	stem_length = scen.world_size * 0.4
	gap = scen.world_size / 30
	half = np.radians(included_deg) / 2.0          # each arm's angle from the stem axis
	left_out = np.array([-np.sin(half), np.cos(half)])
	right_out = np.array([np.sin(half), np.cos(half)])
	for out in (left_out, right_out):
		world.tube_params.append(_tube(scen, merge_point + out * (branch_length + gap),
									   merge_point + out * gap, tube_width))
	world.tube_params.append(_tube(scen, merge_point + np.array([0.0, -gap]),
								   merge_point + np.array([0.0, -(stem_length + gap)]), tube_width))
	avg = np.mean([t['L'] for t in world.tube_params])
	scen.progress_gain = scen.goal_rew / (avg * 10)


class NetworkScenario(_merge.Scenario):
	TUBES = []          # [(entrance (x, y), exit (x, y), width)], km
	ROUTES = []         # [[corridor index, ...]] flown in order
	HANDOFF_AT_GOAL = ()  # corridors whose hand-off waits until the goal past their exit is reached
	ROUTE_CYCLE = None  # order routes are given to aircraft 0, 1, 2, ... (default: round robin)
	SPACING = None      # along-track gap between successive aircraft on one route, km
	LATERAL = 0.3       # +/- lateral spread of the start positions, km

	# ---- geometry -------------------------------------------------------------
	def setup_two_tube_params(self, world):
		world.tube_params = [_tube(self, a, b, w) for a, b, w in self.TUBES]
		avg = np.mean([t['L'] for t in world.tube_params])
		self.progress_gain = self.goal_rew / (avg * 10)

	# ---- start positions and routes ----------------------------------------------
	def assign_routes(self, n):
		cyc = self.ROUTE_CYCLE if self.ROUTE_CYCLE is not None else list(range(len(self.ROUTES)))
		return np.array([cyc[i % len(cyc)] for i in range(n)], dtype=int)

	def spawn_offset(self, route, k):
		"""Distance behind the route's first entrance of the k-th aircraft on it."""
		spacing = self.SPACING if self.SPACING is not None else 6.0 * self.separation_distance
		return (k + 1) * spacing

	def random_scenario(self, world):
		self.setup_two_tube_params(world)
		self._route = self.assign_routes(self.num_agents)
		self._seg = np.zeros(self.num_agents, dtype=int)     # position along the route
		count = np.zeros(len(self.ROUTES), dtype=int)
		for i, agent in enumerate(world.agents):
			r = int(self._route[i])
			first = self.ROUTES[r][0]
			self._seg[i] = 0
			tp = world.tube_params[first]
			k = count[r]; count[r] += 1
			along = -self.spawn_offset(r, k) + np.random.uniform(-0.1, 0.1) * self.separation_distance
			lateral = np.random.uniform(-1.0, 1.0) * self.LATERAL
			agent.state.p_pos = tp['entrance'] + along * tp['e'] + lateral * tp['n']
			heading = np.arctan2(tp['e'][1], tp['e'][0]) + np.random.uniform(-np.pi / 6, np.pi / 6)
			agent.state.reset_velocity(theta=heading)
			agent.state.c = np.zeros(world.dim_c)
			agent.status = False
			self.current_tube[i] = first
		self.agent_id_updated = np.arange(self.num_agents)
		self.landmark_poses = np.zeros((self.num_agents, 2), dtype=np.float32)
		for agent in world.agents:
			self.landmark_poses[self.goal_match_index[agent.id]] = \
				self._goal_pos_for_tube(world.tube_params[self.current_tube[agent.id]])
		self.landmark_poses_occupied = np.zeros(self.num_agents)
		self.landmark_poses_updated = self.landmark_poses.copy()
		if self.max_speed is not None:
			for agent in world.agents:
				self.min_time(agent, world)

	# ---- hand-off at the goal for chosen corridors ------------------------------
	def get_agent_phase(self, agent, world):
		phase = super().get_agent_phase(agent, world)
		# inside reward() the aircraft's corridor index is temporarily remapped
		if getattr(self, "_reward_agent", None) == agent.id:
			real = self._reward_real_tube
		else:
			real = int(self.current_tube[agent.id])
		route = self.ROUTES[int(self._route[agent.id])]
		not_last = int(self._seg[agent.id]) < len(route) - 1
		if real in self.HANDOFF_AT_GOAL and not_last:
			goal = self.landmark_poses[self.goal_match_index[agent.id]]
			if float(np.linalg.norm(agent.state.p_pos - goal)) < self.min_dist_thresh:
				return 2         # reached the goal past this corridor -> hand over
			if phase == 2:
				return 1         # left the corridor but not at the goal yet -> no hand-off
		return phase

	# ---- reward: reuse the merge reward on this aircraft's own route ----------------
	def reward(self, agent, world):
		i = agent.id
		route = self.ROUTES[int(self._route[i])]
		j = int(self._seg[i])
		last = j == len(route) - 1
		real_tubes = list(world.tube_params)
		cur = route[j]
		nxt = route[j] if last else route[j + 1]
		world.tube_params[:] = [real_tubes[cur], real_tubes[cur], real_tubes[nxt]]
		self.current_tube[i] = 2 if last else 0
		self._reward_agent, self._reward_real_tube = i, cur
		try:
			rew = super().reward(agent, world)
		finally:
			self._reward_agent = None
			switched = (not last) and int(self.current_tube[i]) == 2
			if switched:
				self._seg[i] = j + 1
			self.current_tube[i] = route[int(self._seg[i])]
			world.tube_params[:] = real_tubes
		return rew

	def info_callback(self, agent, world):
		info = super().info_callback(agent, world)
		route = self.ROUTES[int(self._route[agent.id])]
		info['On_last_corridor'] = bool(int(self._seg[agent.id]) == len(route) - 1)
		return info
