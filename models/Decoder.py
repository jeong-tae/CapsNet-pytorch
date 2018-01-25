import torch
import torch.nn as nn
import torch.nn.functional as F

#TODO: unit test

class decoder(nn.Module):
	def __init__(self, config, n_cls):
		"""
			Args
				config: It contains network parameter infomation such number of parameter for each layer
		"""
		# 3 fc layter
		self.fc1 = nn.Linear(n_cls * in_units, 512)
		self.fc2 = nn.Linear(512, 1024)
		self.fc3 = nn.Linear(1024, 784) # 784 is size of mnist image

	def forward(self, x, y = None):
		# mask out all but the activity vector of correct digit capsule at training time
		masked = self.mask(x, y)
		
		# reconstruction with fc layers
		recon = nn.ReLU(self.fc1(masked), inplace = True)
		recon = nn.ReLU(self.fc2(recon), inplace = True)
		recon = nn.Sigmoid(self.fc3(recon))

		return recon

	def mask(self, x, y = None):
		"""
			Args
				x: [batch, n_cls, in_units]
				y: [batch, n_cls]
			
			return
				masked: [batch, n_cls, in_units]

		"""
		if y == None:
			# TODO: complete this ft
			raise NotImplemented(" [!] Reconstruction at test tims is not implemented yet")
		else:
			masked = (x * y.view(y.size(0), y.size(1), 1))
			return masked
