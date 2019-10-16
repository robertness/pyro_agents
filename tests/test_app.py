from pyro.infer import Importance
import torch


from pyro_agents.agents import add_factor
from pyro_agents.agents import Agent

import unittest


class TestAgent(unittest.TestCase):
    def setUp(self):
        self.agent = Agent()

    def test_smoke(self):
        agent = Agent()
        action_dist = agent.infer_actions(
            torch.tensor(0.),
            torch.tensor(4.)
        )
        print(action_dist)

    def test_add_factor(self):
        expected_logmass = torch.tensor(-1.0)

        def model():
            add_factor(expected_logmass, 'test')
        posterior = Importance(model, num_samples=1).run()
        actual_logmass = next(posterior._traces())[1]
        self.assertEqual(actual_logmass, expected_logmass)


if __name__ == '__main__':
    unittest.main()
