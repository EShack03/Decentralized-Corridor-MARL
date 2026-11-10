"""[ES] added: sequential (90-degree bend) layout, variant 1 -- intermediate target
at corridor 2's entrance.

Identical to three_phase_graph_sequential.py (geometry, spawning, reward, hand-off
rule, final goal) except for ONE thing: while an aircraft is in corridor 1, the goal
it observes is placed at the entrance of corridor 2 instead of 1.25 km straight past
corridor 1's exit (world_size / 4 along corridor 1's axis, which is 0.42 km beyond
the level of corridor 2).

Purpose: test whether the far-away intermediate target makes aircraft arrive at the
junction at full speed and bunch in the junction gap / early in corridor 2
(diagnostic of 10 Nov 2026: all aircraft reached the junction at 90 m/s; ~1/5 of
unfiltered losses of separation in the gap, ~1/3 early in corridor 2).

Notes:
  * The hand-off to the final goal is unchanged (it still happens when the
    aircraft passes corridor 1's exit gate), so the only difference is where the
    target sits beforehand.
  * Goals are graph nodes in this scenario, so other aircraft also see these
    intermediate goal nodes at corridor 2's entrance (as they saw them past
    corridor 1's exit in the original).
  * Success is unchanged: exiting the final corridor correctly.
The original scenario file is not modified.
"""
import numpy as np

from multiagent.custom_scenarios import three_phase_graph_sequential as _seq


class Scenario(_seq.Scenario):

	def setup_two_tube_params(self, world):
		super().setup_two_tube_params(world)
		self._tp = world.tube_params     # current layout (re-randomised every reset)

	def _goal_pos_for_tube(self, tube):
		tp = getattr(self, "_tp", None)
		if tp is not None and len(tp) > 1 and tube is tp[0]:
			return np.array(tp[1]['entrance'], dtype=float)
		return super()._goal_pos_for_tube(tube)
