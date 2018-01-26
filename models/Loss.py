import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

class marginLoss(nn.Module):
    def __init__(self, p_margin = 0.9, n_margin = 0.1, lamb = 0.5):
        super(marginLoss, self).__init__()
        
        self.p_margin = p_margin
        self.n_margin = n_margin
        self.lamb = lamb

    def forward(self, x, y):
        """
            Args
                x: [batch, 10, 16, 1]
                y: [batch, 10]

            return
                m_loss
        """
        # v_norm: [batch, 10]
        x = x.squeeze(3)

        v_norm = torch.sqrt((x**2).sum(dim = 2, keepdim = False))
        
        m_loss = ((y * F.relu(self.p_margin - v_norm)**2) + (self.lamb * (1. - y) * F.relu(v_norm - self.n_margin)**2))
        m_loss = torch.sum(m_loss, dim = 1).mean() # Margin loss, described in section 3

        return m_loss

class reconstructionLoss(nn.Module):
    def __init__(self, lamb = 0.0005):
        super(reconstructionLoss, self).__init__()
        self.lamb = 0.0005 # scale down

    def forward(self, x, y):
        """
            Args
                x: [batch, 784], 784 is size of sample in MNIST
                y: [batch, 28, 28, 1]

            return
                r_loss
        """
        y = y.view(y.size(0), -1)
        try:
            r_loss = torch.sum((x - y)**2, dim = 1)
        except:
            import pdb
            pdb.set_trace()
        r_loss = r_loss.mean() # Mean squared loss
        return r_loss * self.lamb

if __name__ == '__main__':
    x1 = torch.FloatTensor(2, 10, 16, 1).uniform_(-1, 1)
    x1 = Variable(x1)
    y1 = torch.FloatTensor(2, 10).uniform_(-1, 1)
    y1 = Variable(y1)
    print(" [*] margin loss input:", x1.size())
    margin = marginLoss()
    m_loss = margin(x1, y1)
    print(" [*] m_loss:", m_loss)

    x2 = torch.FloatTensor(2, 784).uniform_(-1, 1)
    x2 = Variable(x2)
    y2 = torch.FloatTensor(2, 1, 28, 28).uniform_(-1, 1)
    y2 = Variable(y2)
    print(" [*] recon loss input:", x2.size())
    recon = reconstructionLoss()
    r_loss = recon(x2, y2)
    print(" [*] r_loss:", r_loss)
