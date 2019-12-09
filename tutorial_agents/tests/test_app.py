import pyro
from pyro.infer import Importance
import torch
from tutorial_agents.agents import add_factor
from tutorial_agents.agents import Agent
import unittest

class TestAgent(unittest.TestCase):
	def setUp(self):
		self.agent = Agent()

	def test_smoke(self):
		"""
		Test Agent class allowing time_left to be 4
		"""
		
		agent = Agent()
		action_dist = agent.infer_actions(
			torch.tensor(0.),
			torch.tensor(4.)
		)
		print('Agent action possible vals: ',action_dist[0].data,
			  '; with prob distribution: ',action_dist[1].data)

	def test_simulate_fct(self):
		"""
		Test to see if simulate() outputs correct trajectory
		Small scale: time_left is 4
		"""
		agent = Agent()
		start_state = torch.tensor(0.)
		total_time = torch.tensor(4.)
		print('Agent state trajectory:', agent.simulate(start_state, total_time).data)

	def test_add_factor(self):
		"""
		Testing output of add_factor() function
		"""
		expected_logmass = torch.tensor(-100.0)
		val = torch.tensor(-1.0)
		def model():
			add_factor(val, 'exp_util_test')  # alpha is 100.0
		posterior = Importance(model, num_samples=1).run()
		actual_logmass = next(posterior._traces())[1]
		self.assertEqual(actual_logmass, expected_logmass)

	def test_action_model_add_logmass(self):
		"""
		Test to see if add_factor() changes logmass
		to desired output with Importance sampling
		"""
		agent = Agent()
		state = torch.tensor(1.0)
		time_left = torch.tensor(0.0)
		dummy_utility = torch.tensor(1.0)

		# Create dummy utility function
		def dummy_utility_func(state, action, time_left):
			return dummy_utility
		# Set expected utility to the dummy function
		agent.expected_utility = dummy_utility_func

		posterior = Importance(
			agent.action_model,
			num_samples=1
		).run(state, time_left)

		actual_logmass = next(posterior._traces(state, time_left))[1]
		expected_logmass = 100 * dummy_utility
		self.assertEqual(actual_logmass, expected_logmass)

	def test_action_model_output(self):
		"""
		Test to see if action_model()
		has desired output
		"""
		agent = Agent()
		state = torch.tensor(1.0)
		time_left = torch.tensor(0.0)
		dummy_utility = torch.tensor(1.0)

		# Create dummy utility function
		def dummy_utility_func(state, action, time_left):
			return dummy_utility

		# Set expected utility to the dummy function
		agent.expected_utility = dummy_utility_func

		expected_action = torch.tensor(0.0)

		actual_action = pyro.condition(
			agent.action_model,
			{'action_test': expected_action}
		)(state, time_left,testing_mode = True)

		self.assertEqual(actual_action, expected_action)


if __name__ == '__main__':
	unittest.main()
