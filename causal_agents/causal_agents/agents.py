import torch
import numpy as np
import pyro
from pyro.infer import EmpiricalMarginal, Importance
import pyro.distributions as dist
import uuid

from causal_agents.environments import Integer_Line_Environment as Environment
from causal_agents.utils import sample_and_count


class Agent():
	"""
	Agent Class
	"""

	def model(self, state, time_left):
		"""
		Agent Model: data generation process
		"""
		action = pyro.sample('action_{}_{}'.format(state,time_left),
							 dist.Categorical(torch.ones(3))
							)
		next_state = self.mental_transition(state,action)
		next_utility = self.utility(next_state)

		return {'action_{}_{}'.format(state,time_left): action,
				'next_state_{}_{}'.format(state,time_left): next_state,
				'next_utility_{}_{}'.format(state,time_left): next_utility}

	def simulate(self, state, time_left, output):
		"""
		Agent Simulation: Forward simulation process
		"""
		action = self.model(state, time_left)['action_{}_{}'.format(state,time_left)]
		interv_model = pyro.do(self.model, data={'action_{}_{}'.format(state,time_left): action})
		next_state = interv_model(state, time_left)['next_state_{}_{}'.format(state,time_left)]
		next_utility = interv_model(state, time_left)['next_utility_{}_{}'.format(state,time_left)]

		if not torch.eq(time_left, torch.tensor(1.)): # if there are moves left
			future_utility_posterior = self.infer(next_state, time_left-1,'exp_utility')
#             print('simulate: posterior',future_utility_posterior, 'state',state,'timeleft',time_left)
			exp_utility = next_utility + torch.mean(future_utility_posterior)
			self.add_factor(exp_utility, 'exp_util_{}_{}'.format(state,time_left))
		else:
			exp_utility = next_utility
			self.add_factor(exp_utility, 'exp_util_{}_{}'.format(state,time_left))

		if output=='action':
			return action
		elif output=='exp_utility':
			return exp_utility

	def play(self, state, time_left):
		"""
		Puts policy into action and moves to next state
		"""
		while not torch.eq(time_left, torch.tensor(0.)):
			action_posterior = self.infer(state, time_left,'action')
			action = self.policy(action_posterior)
			interv_model = pyro.do(self.model, data={'action_{}_{}'.format(state,time_left): action})
			next_state = interv_model(state, time_left)['next_state_{}_{}'.format(state,time_left)]
			
			# Print Trajectory and Actions  
			print('State:',state.item(),
				', Action: ', list(Environment().action_dictionary.keys())[action])
			return action, self.play(next_state, time_left-1)

	def infer(self, state, time_left, variable, n_samples=10):
		"""
		Agent inference method: backwards from the simulation evidence
		"""
		## OR ADD ACTION (as before)???
		param_name = 'posterior_var{}_state{}_time{}'.format(variable, state.int(), time_left.int())

		# if already computed:
		if param_name in list(pyro.get_param_store().keys()):
			variable_posterior = pyro.get_param_store().get_param(param_name)
			return variable_posterior

		else: # else need to compute:
			imp_sampling_posterior = Importance(
				self.simulate,
				num_samples = n_samples
			).run(state,time_left,variable)
			variable_posterior = sample_and_count(imp_sampling_posterior,variable)

			# Save util_param for later
			param = pyro.param(param_name, variable_posterior)
			return variable_posterior

	def add_factor(self, val, name_, alpha = 100):
		"""
		Adds factor value to the unnormalized log-prob
		of the model sampling, to maximize soft max.
		"""
		pyro.sample(
			name_,
			dist.Delta(val, log_density=alpha*val),
			obs=val
		)

	def policy(self, action_posterior):
		"""
		Agent Policy to select action
		"""
		action = pyro.sample(
			'action_policy',
			dist.Categorical(action_posterior)
		)
		return action

	def mental_transition(self, state, action_index):
		"""
		Agent transition same as environment transition
		"""
		return Environment().transition(state, action_index)

	def utility(self, state):
		"""
		Agent utility same as environment utility
		"""
		return Environment().utility(state)