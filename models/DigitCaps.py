import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class digitCaps(nn.Module):
    """

    """
    def __init__(self, in_size, in_units, out_units, n_cls, r_iteration):
        super(digitCaps, self).__init__()

        self.n_cls = n_cls
        self.r_iteration = r_iteration
        self.W = nn.Parameter(1, in_size, n_cls, out_units, in_units)

    def forward(self, u):
    """
        Implementation of routing algorithm

        Args
            u: [batch, in_units, total_caps]

        returns

    """
        # tile the W/u for matrix multiplication
        W_tiled = torch.cat([self.weight] * batch_size, dim = 0)
        u = torch.stack([u] * self.n_cls, dim = 2).unsqueeze(4)

        # (out_units, in_units) * (in_units, 1) matrix mulplication -> (out_units, 1)
        # u_ji is prediction vertor, shape: [batch, in_size, n_cls, out_unit, 1]
        u_ji = torch.matmul(W_tiled, u)

        # rounting logits b_ij are initialized to zero
        # TODO: cuda enable
        b_ij = Variable(torch.zeros(u_ji.size(0), u_ji.size(1), self.n_cls, 1))

    def softmax(self, x, dim = 1):
        """
            pytorch F.softmax doesn't support take a dimension as args
            
            This is temporary expdient till support
        """
        transposed_input = x.transpose(dim, len(x.size()) - 1)
        softmax_out = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size()))
        return softmax_out.view(*transposed_input.size()).transpose(dim, len(x.size()) - 1)
