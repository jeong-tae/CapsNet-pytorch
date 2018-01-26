import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .utils import squash

class digitCaps(nn.Module):
    """

    """
    def __init__(self, in_size, in_units, out_units, n_cls, r_iterations, cuda = False):
        super(digitCaps, self).__init__()

        self.cuda = cuda
        self.n_cls = n_cls
        self.r_iterations = r_iterations
        self.W = nn.Parameter(torch.randn(1, in_size, n_cls, out_units, in_units))

    def forward(self, u):
        """
            Implementation of routing algorithm

            Args
                u: [batch, total_caps, in_units]

            returns
                v_j: [batch, in_size, n_cls, 1]

        """
        # tile the W/u for matrix multiplication
        W_tiled = torch.cat([self.W] * u.size(0), dim = 0)
        u = torch.stack([u] * self.n_cls, dim = 2).unsqueeze(4)

        # (out_units, in_units) * (in_units, 1) matrix mulplication -> (out_units, 1)
        # u_ji is prediction vertor, shape: [batch, in_size, n_cls, out_units, 1]
        u_ji = torch.matmul(W_tiled, u)

        # rounting logits b_ij are initialized to zero
        b_ij = Variable(torch.zeros(u_ji.size(0), u_ji.size(1), self.n_cls, 1))
        if self.cuda:
            b_ij = b_ij.cuda()
        
        for i in range(self.r_iterations):
            c_ij = F.softmax(b_ij, 2).unsqueeze(4) # Eq.3, Coupling coefficient
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
            
if __name__ == '__main__':
    x = torch.ones(2, 1152, 8)
    x = Variable(x)
    print(" [*] input:", x.size())

    net = digitCaps(1152, 8, 16, 10, 3, False)
    capsules = net(x)
    print(" [*] output:", capsules.size())

