import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .DigitCaps import digitCaps
from .PrimaryCaps import primaryCaps

class capsNet(nn.Module):
	def __init__(self, config):
		super(capsNet, self).__init__()
	
		self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 256, kernel_size = 9, stride = 1)
		self.primaryCaps = primaryCaps(in_size = 256, out_size = 32, kernel_size = 9, stride = 2, n_units = 8)
		self.digitCaps = digitCaps(in_size = (6*6*32), in_units = 8, out_units = 16, n_cls = 10, r_iterations = config.r_iterations, cuda = config.cuda)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = self.primaryCaps(x)
		x = self.digitCaps(x)

		return x

if __name__ == '__main__':
	x = torch.FloatTensor(2, 1, 28, 28).uniform_(-1, 1)
	x = Variable(x).cuda()
	print(" [*] input:", x.size())
	class config(object):
		def __init__(self, r_iter, cuda):
			self.r_iterations = r_iter
			self.cuda = cuda
	
	args = config(3, True)

	net = capsNet(args).cuda()
	out = net(x)
	print(" [*] output:", out.size())
