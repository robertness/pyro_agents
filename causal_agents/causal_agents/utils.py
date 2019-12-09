import torch
import numpy as np
import pyro
from pyro.infer import EmpiricalMarginal, Importance

### Helper functions ###

def sample_and_count(imp_sampling_posterior, variable, n_samples=10):
	"""
	Helper function:
	- For variable = 'action': sample from action posterior, count,
	and calculate probs of each value
	- For variable = 'exp_utility', sample from utility posterior,
	and return the mean sample (in noiseless mdp case samples are all the same)
	"""
	# Sample
	marginal_dist = EmpiricalMarginal(imp_sampling_posterior).sample(
		(n_samples,1)
	).float()

	if variable=='action':
		# Initialize
		vals=torch.tensor((0.,1.,2.))
		counts = torch.zeros(vals.shape).float()
		# Count
		for i in range(list(vals.shape)[0]):
			counts[i] = (marginal_dist==vals[i]).sum()
		probs = counts/n_samples
		return probs

	elif variable=='exp_utility':
		return torch.mean(marginal_dist) # assumed to be all the same
