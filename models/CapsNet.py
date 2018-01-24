import torch
import torch.nn as nn
from torch.autograd import Variable

from DigitCaps import digitCaps
from PrimaryCaps import primaryCaps

class capsNet(nn.module):
	def __init__(self, config):
		super(capsNet, self).__init__()

		self.conv1 = nn.Conv2d(in_channel = 1, out_channel = 256, kernel_size = 9, stride = 1)
		self.primaryCaps = primiaryCaps(in_size = 256, out_size = 32, kernel_size = 9, stride = 2, n_units = 8)
		self.digitCaps = digitCaps(in_size = (6*6*32), in_units = 8, out_units = 16, n_cls = 10, r_iteration = 3)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = self.primaryCaps(x)
		x = self.digitCaps(x)

		return x
