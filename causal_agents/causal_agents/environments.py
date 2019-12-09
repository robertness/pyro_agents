import torch
import pyro
import uuid
import pyro.distributions as dist


class Integer_Line_Environment():
	"""
	Environment Class
	"""
	def __init__(self):
		self.action_dictionary = {'Go Left': torch.tensor(-1.),
								'Stay': torch.tensor(0.),
								'Go Right': torch.tensor(1.)}
	
	def transition(self, state, action_index):
		"""
		Given a state and action, transition to the next state.
		Input: - state (integer)
			   - action_index (0,1,2): index of the action value
		Output:- next state (integer)
		"""
		# helper function:
		def add_state_action(state, action_index):
			numeric_action_val = list(self.action_dictionary.values())[action_index.long()]
			s = state.add(numeric_action_val)
			return s

		uid = str(uuid.uuid4())
		S = pyro.sample(
			'next_state_{}'.format(uid), # uid for unique names
			dist.Delta(add_state_action(state, action_index))
		)
		return S

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