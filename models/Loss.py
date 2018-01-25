import torch
import torch.nn.functional as F


class marginLoss(nn.Module):
	def __init__(self, p_margin = 0.9, n_margin = 0.1, lamb = 0.5):
		super(Margin_loss, self).__init__()
		
		self.p_margin = p_margin
		self.n_margin = n_margin
		self.lamb = lamb

	def foward(self, x, y):
		"""
			Args
				x: [batch, 10, 16]
				y: [batch, 10]

			return
				m_loss
		"""
		# v_norm: [batch, 10]
		v_norm = torch.sqrt((x**2).sum(dim = 2, keepdim = False)
		
		m_loss = (y * F.relu(self.p_margin - v_norm)**2) + (self.lamb * (1. - y) * F.relu(v_norm - self.n_margin)**2)
		m_loss = torch.sum(m_loss, dim = 1).mean() # Margin loss, described in section 3

		return m_loss

class reconstructionLoss(nn.Module):
	def __init__(self, lamb = 0.0005):
		super(ReconstructionLoss, self).__init__()
		self.lamb = 0.0005 # scale down

	def foward(self, x, y):
		"""
			Args
				x: [batch, 784], 784 is size of sample in MNIST
				y: [batch, 28, 28, 1]

			return
				r_loss
		"""
		y = y.view(y.size(0), -1)
		r_loss = torch.sum((x - y)**2, dim = 1)
		r_loss = r_loss.mean() # Mean squared loss
		return r_loss * self.lamb
