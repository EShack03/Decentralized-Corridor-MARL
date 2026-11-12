"""[ES] added: sequential (90-degree bend) layout with any combination of three vertiport
settings switched on (diagnostic factorial, 12 Nov 2026). Not run directly: the runnable
layouts are the thin files three_phase_graph_seqmix_<none|g|s|w|gs|gw|sw|gsw>.py.

Question: the vertiport repeats the sequential bend (next entrance 0.83 km across and
0.83 km down) yet scores ~100 % where sequential loses aircraft that turn late and miss
corridor 2's entrance. Which of the three remaining differences fixes that?
  G  goal spacing : goal placed world_size/12 = 0.42 km past each corridor's exit
                    (vertiport, via the merge scenario) instead of world_size/4 = 1.25 km
  S  spawn        : queue 6 x separation = 0.91 km apart, +/-0.3 km sideways
                    (vertiport) instead of 1.2 x 3 x separation = 0.55 km, +/-1.5 km
  W  width        : both corridors 400 m wide (vertiport final approach / bottom
                    corridor) instead of 750 m
Everything else (dynamics, observation, rewards, gate and success rules, corridor
lengths and positions) is the original sequential scenario's. With all three off it
reproduces three_phase_graph_sequential exactly (checked: same positions every step).
Random draws are made in the original order, so a seed keeps its turn direction.
"""
import numpy as np
from multiagent.custom_scenarios import three_phase_graph_sequential as _seq


class Scenario(_seq.Scenario):
	VERT_GOAL = False     # G
	VERT_SPAWN = False    # S
	VERT_WIDTH = False    # W
	WIDTH_KM = 0.4

	# ---- G: goal distance past each corridor's exit -------------------------------
	def _goal_pos_for_tube(self, tube):
		if not self.VERT_GOAL:
			return super()._goal_pos_for_tube(tube)
		return np.array(tube['exit']) + (self.world_size / 12) * np.array(tube['e'])

	# ---- W: corridor width ---------------------------------------------------------
	def setup_two_tube_params(self, world):
		super().setup_two_tube_params(world)
		if self.VERT_WIDTH:
			world.tube_params = [
				self._build_tube_params(t['entrance'], t['exit'], self.WIDTH_KM, t['angle'],
										t['length'], t['rotation_matrix'])
				for t in world.tube_params]

	# ---- S: start positions ---------------------------------------------------------
	def random_scenario(self, world):
		"""Same as the original except, with VERT_SPAWN, the queue spacing and sideways spread."""
		self.setup_two_tube_params(world)
		sep = self.separation_distance
		if self.VERT_SPAWN:
			long_spacing, lateral_spread = 6.0 * sep, 0.3
		else:
			long_spacing, lateral_spread = 1.2 * max(3.0 * sep, 1.5 * sep), self.world_size * 0.3
		current_tube = world.tube_params[0]
		for i in range(self.num_agents):
			self.current_tube[i] = 0
			corridor_e, corridor_n, entrance = current_tube['e'], current_tube['n'], current_tube['entrance']
			along_offset = -(i + 1) * long_spacing
			lateral_jitter = np.random.uniform(-1.0, 1.0) * lateral_spread
			along_jitter = np.random.uniform(-0.2, 0.2) * long_spacing
			random_pos = entrance + (along_offset + along_jitter) * corridor_e + lateral_jitter * corridor_n
			agent = world.agents[i]
			if self.is_obstacle_collision(random_pos, agent.size, world):
				lateral_jitter = np.random.uniform(-1.0, 1.0) * lateral_spread * 0.5
				random_pos = entrance + (along_offset + along_jitter) * corridor_e + lateral_jitter * corridor_n
			agent.state.p_pos = random_pos
			corridor_heading = np.arctan2(corridor_e[1], corridor_e[0])
			agent.state.reset_velocity(theta=corridor_heading + np.random.uniform(-np.pi / 6, np.pi / 6))
			agent.state.c = np.zeros(world.dim_c)
			agent.status = False
		self.agent_id_updated = np.arange(self.num_agents)
		for agent in world.agents:
			lm = world.landmarks[self.goal_match_index[agent.id]]
			lm.state.p_pos = self._goal_pos_for_tube(world.tube_params[self.current_tube[agent.id]])
			lm.state.reset_velocity()
		self.landmark_poses = np.array([lm.state.p_pos for lm in world.landmarks])
		self.landmark_poses_occupied = np.zeros(self.num_agents)
		self.landmark_poses_updated = np.array([lm.state.p_pos for lm in world.landmarks])
		self.agent_id_updated = np.arange(self.num_agents)
		if self.max_speed is not None:
			for agent in world.agents:
				self.min_time(agent, world)
