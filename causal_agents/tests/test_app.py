import pyro
from pyro.infer import Importance
import torch
from causal_agents.agents import Agent
import unittest

class TestAgent(unittest.TestCase):
	def setUp(self):
		self.agent = Agent()

	def test_smoke(self):
		"""
		Test Agent class allowing time_left to be 4
		"""
		init_state = torch.tensor(0.0)
		total_time = torch.tensor(4.0)
		print('Agent state trajectory and actions:')
		Agent().play(init_state, total_time)
		pyro.clear_param_store()


if __name__ == '__main__':
	unittest.main()
