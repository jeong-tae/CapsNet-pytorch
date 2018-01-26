import torch
import torch.nn.functional as F

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
	v_j = scale * s_j / torch.sqrt(norm)
	return v_j

def accuracy(x, y):
	"""
		Prediction is vector.
		Length of the vector represent the existence of entity.
		Args
			x: [batch, n_cls, n_units]
			y: [batch, n_cls] ??
		return
			acc: float
	"""
	x_len = torch.sqrt((x**2).sum(dim = 2))
	pred = F.softmax(x_len, dim = 1)
	
	_, idx = pred.max(dim = 1)
	# TODO: is y vector? or scalar?
	
	correction = torch.eq(idx, y)
	acc = correction.mean()
	return acc
