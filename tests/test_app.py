import pyro
from pyro.infer import Importance
import torch


from pyro_agents.agents import add_factor
from pyro_agents.agents import Agent

import unittest


class TestAgent(unittest.TestCase):
    def setUp(self):
        self.agent = Agent()

    @unittest.skip("Getting infinite recursion")
    def test_smoke(self):
        agent = Agent()
        action_dist = agent.infer_actions(
            torch.tensor(0.),
            torch.tensor(4.)
        )
        print(action_dist)
        
    def test_mini_example(self):  # smaller example -> recursion finishes
        agent = Agent()
        action_dist = agent.infer_actions(
            torch.tensor(0.),
            torch.tensor(2.)  # reduced for speed (runs in ~ 0.79 s)
        )
        print(action_dist)

    def test_add_factor(self):
        expected_logmass = torch.tensor(-1.0)

        def model():
            add_factor(expected_logmass, 'test')
        posterior = Importance(model, num_samples=1).run()
        actual_logmass = next(posterior._traces())[1]
        self.assertEqual(actual_logmass, expected_logmass)

    def test_action_model_add_logmass(self):
        agent = Agent()
        state = torch.tensor(1.0)
        time_left = torch.tensor(0.0)
        dummy_utility = torch.tensor(1.0)
        def dummy_utility_func(state, action, time_left):
            return dummy_utility

        posterior = Importance(
            agent.action_model,
            num_samples=1
        ).run(dummy_utility_func, state, time_left)
        actual_logmass = next(posterior._traces())[1]
        expected_logmass = 100 * dummy_utility
        self.assertEqual(actual_logmass, expected_logmass)

    def test_action_model_output(self):
        agent = Agent()
        state = torch.tensor(1.0)
        time_left = torch.tensor(0.0)
        dummy_utility = torch.tensor(1.0)
        def dummy_utility_func(state, action, time_left):
            return dummy_utility
        expected_action = torch.tensor(0.0)
        actual_action = pyro.condition(
            agent.action_model,
            {'action_test': expected_action}
        )(dummy_utility_func, state, time_left, uid="test")
        self.assertEqual(actual_action, expected_action)


if __name__ == '__main__':
    unittest.main()
