"""[ES] added: sequential (90-degree bend) layout with 400 m wide corridors
(variant, 12 Nov 2026).

Identical to three_phase_graph_sequential.py (corridor positions and lengths,
random left/right turn, spawning, reward, hand-off rule, goals, success rule)
except for ONE thing: both corridors are 400 m wide instead of 750 m (the original
width = max(3 * aircraft size * 2.5, world_size * 0.15) = 0.75 km). 400 m is the
width of the vertiport's final-approach and bottom corridors, whose junction copies
the sequential junction geometry (next corridor 0.833 km across and down) and keeps
~99.5 % success with the frozen filter even at 40-50 aircraft.

Everything that the scenario derives from a corridor's width follows the new width
(in-tube and gate tests, formation lines), because the corridors are rebuilt with
the scenario's own _build_tube_params. The random draws are unchanged, so a given
seed gives the same turn direction and the same start positions as the original.
Question: does a narrower corridor (less sideways spread entering the turn) explain
why the vertiport's identical junction performs better, and what does it cost in
throughput and completion time? The original scenario file is not modified.
"""
from multiagent.custom_scenarios import three_phase_graph_sequential as _seq


class Scenario(_seq.Scenario):
	CORRIDOR_WIDTH = 0.4      # km (original 0.75)

	def setup_two_tube_params(self, world):
		super().setup_two_tube_params(world)
		world.tube_params = [
			self._build_tube_params(t['entrance'], t['exit'], self.CORRIDOR_WIDTH, t['angle'],
									t['length'], t['rotation_matrix'])
			for t in world.tube_params
		]
