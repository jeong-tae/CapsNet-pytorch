import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

class decoder(nn.Module):
    def __init__(self, config):
        super(decoder, self).__init__()
        """
            Args
                config: It contains network parameter infomation such number of parameter for each layer
        """

        self.n_cls = config.num_classes

        # 3 fc layter
        self.fc1 = nn.Linear((self.n_cls * 16), 512) # 16 means in_units
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 784) # 784 is size of mnist image


    def forward(self, x, y = None):
        # mask out all but the activity vector of correct digit capsule at training time
        masked = self.mask(x, y).view(x.size(0), -1)
        
        # reconstruction with fc layers
        recon = F.relu(self.fc1(masked), inplace = True)
        recon = F.relu(self.fc2(recon), inplace = True)
        recon = F.sigmoid(self.fc3(recon))

        return recon

    def mask(self, x, y = None):
        """
            Args
                x: [batch, n_cls, in_units, 1]
                y: [batch, n_cls]
            
            return
                masked: [batch, n_cls, in_units]

        """
        x = x.squeeze(3)

        if y is None:
            # TODO: complete this ft
            raise NotImplemented(" [!] Reconstruction at test times is not implemented yet")
        else:
            masked = (x * y.view(y.size(0), y.size(1), 1))
            return masked

if __name__ == '__main__':
    x = torch.FloatTensor(2, 10, 16, 1).uniform_(-1, 1)
    x = Variable(x)
    y = torch.eye(2, 10)
    y = Variable(y)
    print(" [*] input:", x.size())

    class config(object):
        def __init__(self, n_cls):
            self.num_classes = n_cls

    args = config(10)

    decode = decoder(args)
    out = decode(x, y)
    print(" [*] output:", out.size())


