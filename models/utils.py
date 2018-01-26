import torch
import torch.nn.functional as F
from torch.autograd import Variable

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
            x: [batch, n_cls, n_units, 1]
            y: [batch, n_cls] ??
        return
            acc: float
    """
    x = x.squeeze(3)

    x_len = torch.sqrt((x**2).sum(dim = 2))
    pred = F.softmax(x_len, dim = 1)
    
    _, idx = pred.max(dim = 1)
    _, label = y.max(dim = 1)
    
    correction = torch.eq(idx, label)
    acc = correction.float().mean()
    return acc

if __name__ == '__main__':
    x = torch.FloatTensor(2, 10, 16, 1).uniform_(-1, 1)
    x = Variable(x)
    y = torch.eye(2, 10)
    y = Variable(y)

    acc = accuracy(x, y)
    print(" [*] acc: %.4f"%acc)

