"""[ES] added: merge layout with unbalanced, bunched demand (scenario N).

Identical to three_phase_graph_merge.py (geometry, reward, success rule) except for
which arm each aircraft uses and where it starts:
  * HEAVY_SHARE (80 %) of the aircraft use the left arm, the rest the right arm
    (the original alternates arms, 50/50).
  * On the heavy arm aircraft arrive in bunches of BUNCH (3): BUNCH_SPACING apart
    within a bunch, with BUNCH_GAP between bunches. The light arm keeps an even
    spacing of LIGHT_SPACING (the original merge's per-arm spacing).
  * Lateral start spread is LATERAL (narrower than the original's +/-1.5 km so the
    bunches actually arrive together).
The total number of aircraft is unchanged, so results compare with the original
merge at the same N. Questions it answers: does the light arm get starved or the
heavy arm queue up, and does the safety filter cope with platoons arriving at the
merge together. The original scenario file is not modified.
"""
import numpy as np

from multiagent.custom_scenarios import three_phase_graph_merge as _merge


class Scenario(_merge.Scenario):
	HEAVY_SHARE = 0.8
	BUNCH = 3
	LATERAL = 0.3                     # km

	def random_scenario(self, world):
		self.setup_two_tube_params(world)
		sep = self.separation_distance
		bunch_spacing = 3.0 * sep         # 0.46 km (~5 s at 90 m/s)
		bunch_gap = 20.0 * sep            # 3.0 km between bunches
		light_spacing = 12.0 * sep        # 1.83 km = the original merge's per-arm spacing
		n_heavy = int(round(self.HEAVY_SHARE * self.num_agents))
		k_heavy = k_light = 0
		for i, agent in enumerate(world.agents):
			heavy = i < n_heavy
			arm = 0 if heavy else 1
			tp = world.tube_params[arm]
			if heavy:
				b, j = divmod(k_heavy, self.BUNCH)
				along = -(bunch_gap * b + bunch_spacing * (j + 1))
				k_heavy += 1
			else:
				along = -light_spacing * (k_light + 1)
				k_light += 1
			along += np.random.uniform(-0.1, 0.1) * sep
			lateral = np.random.uniform(-1.0, 1.0) * self.LATERAL
			agent.state.p_pos = tp['entrance'] + along * tp['e'] + lateral * tp['n']
			heading = np.arctan2(tp['e'][1], tp['e'][0]) + np.random.uniform(-np.pi / 6, np.pi / 6)
			agent.state.reset_velocity(theta=heading)
			agent.state.c = np.zeros(world.dim_c)
			agent.status = False
			self.current_tube[i] = arm
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
