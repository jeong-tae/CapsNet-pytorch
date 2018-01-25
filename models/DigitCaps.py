import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import squash

# TODO: unit test

class digitCaps(nn.Module):
    """

    """
    def __init__(self, in_size, in_units, out_units, n_cls, r_iterations):
        super(digitCaps, self).__init__()

        self.n_cls = n_cls
        self.r_iterations = r_iterations
        self.W = nn.Parameter(1, in_size, n_cls, out_units, in_units)

    def forward(self, u):
    """
        Implementation of routing algorithm

        Args
            u: [batch, in_units, total_caps]

        returns

    """
        # tile the W/u for matrix multiplication
        W_tiled = torch.cat([self.W] * batch_size, dim = 0)
        u = torch.stack([u] * self.n_cls, dim = 2).unsqueeze(4)

        # (out_units, in_units) * (in_units, 1) matrix mulplication -> (out_units, 1)
        # u_ji is prediction vertor, shape: [batch, in_size, n_cls, out_units, 1]
        u_ji = torch.matmul(W_tiled, u)

        # rounting logits b_ij are initialized to zero
        # TODO: cuda enable
        b_ij = Variable(torch.zeros(u_ji.size(0), u_ji.size(1), self.n_cls, 1))
		
		for i in range(self.r_iterations):
			c_ij = self.softmax(b_ij, 2).unsqueeze(4) # Eq.3, Coupling coefficient
			# c_ij: [batch, in_size, n_cls, 1, 1]
			# sum of capsules(dim=1)
			s_j = (c_ij * u_ji).sum(dim = 1, keepdim = True)

			# s_j: [batch, 1, n_cls, out_units, 1]
			v_j = squash(s_j)
			
			# Agreement is dot product of u(below layer output) with v(output of capsule)
			# tile the v_j for matrix multiplication
			v_j_tiled = torch.cat([v_j] * c_ij.size(1), dim = 1)

			# u_ji: [batch, in_size, n_cls, out_units, 1]
			# v_j_tile: [batch, in_size, n_cls, out_units, 1]
			# [1, out_units] * [out_units, 1] -> [1, 1]
			a_ij = torch.matmul(u_ji.transpose(3, 4), v_j).squeeze(4)
			# a_ij: [batch, in_size, n_cls, 1]
			
			# update logit
			b_ij = b_ij + a_ij

		return v_j.squeeze(1) # [batch, in_size, n_cls, 1]
			

    def softmax(self, x, dim = 1):
        """
            pytorch F.softmax doesn't support take a dimension as args
            
            This is temporary expdient till support
        """
        transposed_input = x.transpose(dim, len(x.size()) - 1)
        softmax_out = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)))
        return softmax_out.view(*transposed_input.size()).transpose(dim, len(x.size()) - 1)
