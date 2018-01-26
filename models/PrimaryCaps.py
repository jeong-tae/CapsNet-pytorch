import torch
import torch.nn as nn
from torch.autograd import Variable

from utils import squash

class primaryCaps(nn.Module):
    """
        This layer is implementation of PrimaryCaps that described in Figure1.
        The forward function returns [32 x 6 x 6] capsules with 8D vector for each output.

        No routing is used between Conv1 and PrimaryCapsules in MNIST data.
    """
    def __init__(self, in_size = 256, out_size = 32, kernel_size = 9, 
            stride = 2, n_units = 8):
        super(primaryCaps, self).__init__()

        self.capsules = nn.ModuleList([nn.Conv2d(in_size, out_size, kernel_size,
                    stride) for _ in range(n_units)])

    def forward(self, x):
        """
            Args
                x:

            return
                capsules:
        """
        units = [conv_unit(x) for conv_unit in self.capsules]
        # output shape: [batch_size, n_units, out_size, feature_size, feature_size]
        units = torch.stack(units, dim = 1).view(x.size(0), len(units), -1)
        capsules = squash(units, dim = 2) # [batch, n_units, out_size * feature_size**2]

        return capsules.permute(0, 2, 1)

if __name__ == '__main__':
    x = torch.ones(2, 256, 28, 28)
    x = Variable(x).cuda()
    print(" [*] input:", x.size())

    net = primaryCaps().cuda()
    capsules = net(x)
    print(" [*] output:", capsules.size())
