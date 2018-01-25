import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
from torch.autograd import Variable

from models.CapsNet import capsNet
from models.Loss import marginLoss, reconstructionLoss

import argparse

#TODO: take hyper-params as args and put into model config
parser = argparse.ArgumentParser(description = "Training arguments for CapsNet")
parser.add_argment('--max_epoch', type = int, default = 10, help = "Maximum training epochs")
parser.add_argment('--batch_size', type = int, default = 64, help = "Mini-batch size per iteration")
parser.add_argment('--num_classes', type = int, default = 10, help = "Number of labels")
parser.add_argment('--r_iteration', type = int, default = 3, help = "Routing iteration")
parser.add_argment('--cuda', action = "store_true", default = False, help = "Use cuda to train")

args = parser.parse_args()

#TODO: cuda possible

if args.cuda and torch.cuda.is_available():
	torch.set_default_tensor_type('torch.cuda.FloatTensor')
	print(" [*] Set Cuda: True")
else:
	torch.set_default_tensor_type('torch.FloatTensor')
	print(" [*] Set Cuda: False")

net = capsNet()

if args.cuda:
	net = torch.nn.DataParallel(net)
	cudn.benchmark = True

margin_loss = marginLoss()
recon_loss = reconstructionLoss()

optimizer = optim.Adam(net.parameters())

#TODO: data load
