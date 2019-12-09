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
    ave = np.array(np.average(vals.detach().cpu().numpy(), 
                              weights=probs.detach().cpu().numpy()))
    ave = torch.from_numpy(ave).type(dtype=torch.float)
    return ave


def infer_probs(posterior, possible_vals, n_samples=10):
    """
    Sample from a posterior and calculate probabilities of each value.
    Input: - Pyro posterior distribution
           - n_samples
           - possible vals (for actions: 0,1,2; for utility: 0,1)
    Output:- Values of the samples
           - Prob. of each sample occuring in n_samples
    """
    # Initialize
    vals = possible_vals
    counts = torch.zeros(possible_vals.shape).float()
    
    # Sample
    marginal_dist = EmpiricalMarginal(posterior).sample(
        (n_samples,1)
    ).float()
    
    # Count
    for i in range(len(possible_vals)):
        counts[i] = (marginal_dist == vals[i]).sum()
    probs = counts/n_samples
    
    return vals, probs


class Agent():
    """
    Agent class.
    """
    
    def utility(self, state, desired_state = torch.tensor(3.)):
        """
        Returns utility value.
        Per the problem statement:
            - state=3 returns utility=1
            - any other state returns utility=0
        """
        if torch.eq(state, desired_state): 
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
        
        uid = str(uuid.uuid4()) 
        S = pyro.sample(
            'next_state_{}'.format(uid), # uid for unique names
            dist.Delta(add_state_action(state, action_index))
        )
        return S

    def action_model(self, state, time_left, testing_mode = False):
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
        if testing_mode == False:
            _name = '_state{}_timeleft{}'.format(state, time_left)
        else:  # For testing, use the same state name
            _name = '_test' 
        
        action = pyro.sample(  
            'action'+_name,
            dist.Categorical(torch.ones(3))
        )
        
        eu_val = self.expected_utility(
            state,
            action,
            time_left
        )
        add_factor(eu_val, 'exp_util'+_name)
        
        return action 

    def infer_actions(self, state, time_left, n_samples = 10):
        """
        Infers the probability of each action maximizing utility
        Input: - current state
               - time left
        Output:- values and probabilities of each action maximizing utility
        """
        action_param_name = 'actions_values_at_state_{}_time{}'.format(state.int(), time_left.int())
        
        # already computed:
        if action_param_name in list(pyro.get_param_store().keys()):
            action_param = pyro.get_param_store().get_param(action_param_name)
            vals = action_param[0]
            probs = action_param[1]
            return vals, probs
            
        else: # need to compute:    
            action_posterior = Importance(self.action_model, num_samples = n_samples).run(
                state,
                time_left
            )
            vals, probs = infer_probs(action_posterior, possible_vals = torch.tensor([0.,1.,2.]))
            # Save for later
            action_param = pyro.param(action_param_name,
                                    torch.cat((vals.unsqueeze(0), 
                                               probs.unsqueeze(0)),
                                              dim=0)
                                   )
            return vals, probs

    def infer_utility(self, state, action, time_left, n_samples = 10):
        """
        Infers value and probability of utility.
        Input: - current state, action, time_left
        Output:- values and probabilities each utility being 1
        Calls: utility_model()
        """    
        # Get param name:
        util_param_name = 'util_state{}_action{}_time{}'.format(state.int(), action.int(), time_left.int()) 
        
        # if already computed:
        if util_param_name in list(pyro.get_param_store().keys()): 
            util_param = pyro.get_param_store().get_param(util_param_name)
            vals = util_param[0]
            probs = util_param[1]
            return vals, probs
           
        else: # else need to compute:  
            utility_posterior = Importance(
                self.utility_model,
                num_samples = n_samples
            ).run(state, action, time_left)
            
            vals, probs = infer_probs(utility_posterior, possible_vals = torch.tensor([0.,1.]))

            # Save util_param for later
            util_param = pyro.param(util_param_name,
                                    torch.cat((vals.unsqueeze(0), 
                                               probs.unsqueeze(0)),
                                              dim=0)
                                   )
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
        next_state = self.transition(state, action)
        actions, action_probs = self.infer_actions(next_state, time_left)
        next_action_idx = pyro.sample(
            'next_action_state{}_timeleft{}'.format(state, time_left),
            dist.Categorical(action_probs)
        )
        next_action = actions[next_action_idx]
        exp_utility = self.expected_utility(
            next_state,
            next_action,
            time_left
        )
        return exp_utility
    
    def simulate(self, state, time_left):
        """
        Updates and stores the world state 
        in response to the agent’s actions.
        Input: current state and time left
        Output: trajectory of states
        """
        if torch.eq(time_left, torch.tensor(0.)):
            return torch.Tensor()
        else:
            actions, action_probs = self.infer_actions(state, time_left)
            current_action_idx = pyro.sample(
                'next_action_idx_state{}_timeleft{}'.format(state, time_left),
                dist.Categorical(action_probs)
            )
            current_action = actions[current_action_idx]
            next_state = self.transition(state, current_action)
            
            return torch.cat((state.unsqueeze(0),
                              self.simulate(next_state, time_left-1)))    
