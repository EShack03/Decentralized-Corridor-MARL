"""[ES] added: merge layout with the two arms 180 degrees apart (scenario I, steep-merge
test: the existing 90-degree merge as the baseline and 180 degrees as the extreme).

Identical to three_phase_graph_merge.py (widths, lengths, junction gap, spawning,
reward, success rule) except for the angle between the two arms. The original merge
has its arms 90 degrees apart, so the 90-degree file reproduces it exactly (checked:
identical metrics, seed 1) and is the control. At 180 degrees the arms meet head-on
(a T-junction): their exits are 330 m apart and point straight at each other, and
every aircraft turns 90 degrees into the stem. Approach design normally joins traffic
at about 30 degrees and never above 45, so 180 degrees is a stress test, not a
typical case. The original scenario file is not modified.
"""
from multiagent.custom_scenarios import three_phase_graph_merge as _merge
from multiagent.custom_scenarios.three_phase_graph_network_base import set_merge_tubes


class Scenario(_merge.Scenario):
	INCLUDED_DEG = 180

	def setup_two_tube_params(self, world):
		set_merge_tubes(self, world, self.INCLUDED_DEG)
