import torch

def squash(s_j, dim = 2):
	"""
		Non-linear activation function
		Args
			s_j:
			dim:

		return 
	"""
	norm = (s_j**2).sum(dim = dim, keepdim = True)
	scale = norm / (1 + norm)
	v_j = scale * norm / torch.sqrt(norm)
	return v_j
