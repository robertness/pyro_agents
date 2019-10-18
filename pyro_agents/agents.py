import torch
import numpy as np
import pyro
from pyro.infer import EmpiricalMarginal, Importance
import pyro.distributions as dist
import uuid


def add_factor(val, name_):
    pyro.sample(
        name_,
        dist.Delta(val, log_density=val),
        obs=val
    )


def expectation(vals, probs):
    """
    Perform weighted averages of the utilities
    """
    ave = np.average(vals, weights=probs)
    ave = torch.from_numpy(ave).type(dtype=torch.float)
    return ave


def get_samples(posterior, n_samples=1000):
    marginal_dist = EmpiricalMarginal(posterior).sample(
        (n_samples, 1)
    )
    vals, counts = np.unique(marginal_dist, return_counts=True)
    vals = torch.from_numpy(vals).type(dtype=torch.float)
    probs = torch.from_numpy(counts).type(dtype=torch.float)/n_samples
    return vals, probs


class Agent():
    def utility(self, state):
        """
        Per the problem statement:
        The utility is 1 for the state
        corresponding to the integer 3
        and is 0 otherwise.
        """
        if state == torch.tensor(3.):
            return torch.tensor(1.)
        else:
            return torch.tensor(0.)

    def transition(self, state, action):
        def add_state_action(state, action_index):
            # action_index has values 0,1,2
            # Go Left means: state - 1
            # Stay means: state + 0
            # Go Right means: state + 1
            actions = {'Left': torch.tensor(-1.),
                       'Stay': torch.tensor(0.),
                       'Right': torch.tensor(1.)}
            action = list(actions.values())[action_index]
            s = state.add(action)
            return s

        S = pyro.sample(
            'next_state',
            dist.Delta(add_state_action(state, action))
        )
        return S

    def action_model(self, expected_utility_func, state, time_left, uid=str(uuid.uuid4())):
        print('action model call: {}'.format(uid))
        print('time left: ' + str(time_left))
        action = pyro.sample(
            'action_{}'.format(uid),
            dist.Categorical(torch.ones(3))
        )
        eu_val = expected_utility_func(
            state,
            action,
            time_left
        )
        alpha = 100
        add_factor(alpha * eu_val, 'exp_util_{}'.format(uid))
        return action

    def infer_actions(self, state, time_left):
        action_posterior = Importance(self.action_model, num_samples=1000).run(
            self.expected_utility,
            state,
            time_left
        )
        vals, probs = get_samples(action_posterior)
        return vals, probs

    def infer_utility(self, state, action, time_left):
        print('infer_utility call')
        print('time left: ' + str(time_left))
        utility_posterior = Importance(
            self.utility_model,
            num_samples=1000
        ).run(state, action, time_left)
        vals, probs = get_samples(utility_posterior)
        return vals, probs

    def expected_utility(self, state, action, time_left):
        print('expected_utility call')
        print('time left: ' + str(time_left))
        u = self.utility(state)
        new_time_left = max(time_left - 1, torch.tensor(0.))

        if new_time_left == 0:
            return u
        utility_posterior = self.infer_utility(
            state,
            action,
            new_time_left
        )
        return u + expectation(utility_posterior)

    def utility_model(self, state, action, time_left):
        uid = str(uuid.uuid4())
        print('utility model call: {}'.format(uid))
        print('time left: ' + str(time_left))
        next_state = self.transition(state, action)
        actions, action_probs = self.infer_actions(next_state, time_left)
        next_action_idx = pyro.sample(
            'next_action_idx{}'.format(uid),
            dist.Categorical(action_probs)
        )
        next_action = actions[next_action_idx]
        exp_utility = self.expected_utility(
            next_state,
            next_action,
            time_left
        )
        return exp_utility
