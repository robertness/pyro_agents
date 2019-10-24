import torch
import numpy as np
import pyro
from pyro.infer import EmpiricalMarginal, Importance
import pyro.distributions as dist
import uuid


def add_factor(val, name_, alpha = 100):
    """
    Adds factor value to the unnormalized log-prob
    of the model sampling, to maximize soft max.
    """
    pyro.sample(
        name_,
        dist.Delta(val, log_density=alpha*val),
        obs=val
    )

    
def expectation(vals, probs):
    """
    Expectation of the posterior utilites.
    Input:  - utility values, 
            - prob of each obtaining utility=1
    Output: - weighted average (expectation)

    """
    ave = np.array(np.average(np.array(vals), weights=np.array(probs)))
    ave = torch.from_numpy(ave).type(dtype=torch.float)
    return ave


def get_samples(posterior, n_samples=10):
    """
    Sample from a posterior.
    Input: - Pyro posterior distribution
           - n_samples
    Output:- Values of the samples
           - Prob. of each sample occuring in n_samples
    """
    marginal_dist = EmpiricalMarginal(posterior).sample(
        (n_samples,1)
    )
    vals, counts = np.unique(marginal_dist, return_counts=True)
    vals = torch.from_numpy(vals).type(dtype=torch.float)
    probs = torch.from_numpy(counts).type(dtype=torch.float)/n_samples
    return vals, probs


class Agent():
    """
    Agent class.
    """
    
    def utility(self, state):
        """
        Returns utility value.
        Per the problem statement:
            - state=3 returns utility=1
            - any other state returns utility=0
        """
        if state == torch.tensor(2.):  ###### Reduced for speed ######
            return torch.tensor(1.)
        else:
            return torch.tensor(0.)

    def transition(self, state, action_index):
        """
        Given a state and action, transition to the next state.
        Input: - state (integer)
               - action_index (0,1,2): index of the action value
        Output:- next state (integer)
        """
        def add_state_action(state, action_index):
            actions = {'Left': torch.tensor(-1.),
                       'Stay': torch.tensor(0.),
                       'Right': torch.tensor(1.)}
            action = list(actions.values())[action_index.long()]
            s = state.add(action)
            return s
        
        S = pyro.sample(
            'next_state',
            dist.Delta(add_state_action(state, action_index))
        )
        return S

    def action_model(self, state, time_left):
        """
        Samples actions based that maximize softmax
        of extected utility. 
        Input: - current state
               - time left
        Output:- Softmax distribution over possible future actions. 
        
        Calls self.expected_utility() to evaluate the 
        expectation of sampled actions in that state
        Pyro states: action, expected utility
        """
        uid = str(uuid.uuid4())
        action = pyro.sample(
            'action_{}'.format(uid),
            dist.Categorical(torch.ones(3))
        )
        eu_val = self.expected_utility(
            state,
            action,
            time_left
        )
        add_factor(eu_val, 'exp_util_{}'.format(uid)) #### Q: Does adding factor here REALLY do anything?
        return action 

    def infer_actions(self, state, time_left):
        """
        Infers the probability of each action maximizing utility
        Input: - current state
               - time left
        Output:- values and probabilities of each action maximizing utility
        """
        action_posterior = Importance(self.action_model, num_samples = 5).run(
            state,
            time_left
        )
        vals, probs = get_samples(action_posterior)
        return vals, probs

    def infer_utility(self, state, action, time_left):
        """
        Infers value and probability of utility.
        Input: - current state, action, time_left
        Output:- values and probabilities each utility being 1
        Calls: utility_model()
        """
        utility_posterior = Importance(
            self.utility_model,
            num_samples=6
        ).run(state, action, time_left)
        
        vals, probs = get_samples(utility_posterior) 
        return vals, probs

    def expected_utility(self, state, action, time_left):
        """
        Calculated the expected utility value over future actions.
        Input: - current state, action, time_left
        Output:- current + expected utility of agent in this 
                situation, given all possible future situations
        Calls: infer_utility()
        """
        u = self.utility(state)
        new_time_left = time_left - 1
        if torch.eq(new_time_left, torch.tensor(0.)):
            return u
        else:
            posterior_vals, posterior_probs = self.infer_utility(
                state,
                action,
                new_time_left
            )
            return u + expectation(posterior_vals, posterior_probs)

    def utility_model(self, state, action, time_left):
        """
        For a given scenario, sample next action and its 
        respective utility to infer the total expected utility
        of the scenario.
        Input: - current state, action, time_left
        Output: - the expected utility of the agent
        
        Calls:  - expected_utility() to infer the exp
        utility of the agent
                - infer_actions() in a mutual recursion, 
        bottoming out when a terminal state is 
        reached or when time runs out (in expected_utility())

        """
        uid = str(uuid.uuid4())
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
