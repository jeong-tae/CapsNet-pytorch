import torch
import torch.nn as nn
import torch.nn.functional as F

class decoder(nn.Module):
	def __init__(self, config, n_cls):
		"""
			Args
				config: It contains network parameter infomation such number of parameter for each layer
		"""

		# 3 fc layter
