import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
from torch.autograd import Variable

from models.CapsNet import capsNet
from models.Loss import marginLoss, reconstructionLoss
from data.DataLoader import load_mnist

import argparse

#TODO: take hyper-params as args and put into model config
parser = argparse.ArgumentParser(description = "Training arguments for CapsNet")
parser.add_argment('--max_iter', type = int, default = 100000, help = "Maximum training iterations")
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

# TODO: weight load
net = capsNet()

if args.cuda:
	net = torch.nn.DataParallel(net)
	cudn.benchmark = True

margin_loss = marginLoss()
recon_loss = reconstructionLoss()

optimizer = optim.Adam(net.parameters())
print(" [*] Training is ready now!")
# TODO: set logger

def train():
	net.train()
	m_loss = 0
	r_loss = 0
	epoch = 0

	# TODO: data load
	epoch_size = len(dataset) // args.batch_size
	steps = 0

	print(" [*] Training on MNIST dataset")
	batch_iterator = None
	trainset, train_loader = load_mnist(n_worker = 4, batch_size = args.batch_size, split = 'train')
	testset, test_loader = load_mnist(n_worker = 4, batch_size = args.batch_size, split = 'test')

	for iteration in range(0, args.max_iter):
		if (not batch_iterator) or (iteration % epoch_size == 0):
			# create
			batch_iteration = iter(train_loader)
		# TODO?: lr decay??

		images, targets = next(batch_iterator)

		# TODO: cuda enable
		images = Variable(images)
		targets = Variable(targets, volatile = True)

		t0 = time.time()
		out = net(images)
		opt.zero_grad()
		m_loss, r_loss = margin_loss(out, targets), recon_loss(out, images)
		loss = (m_loss + r_loss)
		loss.backward()
		opt.step()
		t1 = time.time()

		#================ TensorBoard logging ================#

		if (iteration % 10) == 0: # Display period
			print(' [*] Iter %d || Loss: %.4f || m_loss: %.4f || r_loss: %.4f || Timer: %.4f sec'%(iteration, loss.data[0], m_loss.data[0], r_loss.data[0], (t1 - t0)))

		if (iteration % 200) == 0: # Eval period
			# set to net eval mode
			net.eval()
			test_iteration = iter(test_loader)

			test_loss = []
			for i in range(0, int(len(testset)/args.batch_size)):
				test_images, test_targets = next(test_iteration)
				# TODO:cuda enble
				test_images = Variable(test_images)
				test_targets = Variable(test_targets, volatile = True)

				out = net(test_images)
				m_loss, r_loss = margin_loss(out, test_targets), recon_loss(out, test_images)
				loss = (m_loss.data[0], r_loss.data[0])
				test_loss.append(loss)
			test_loss = np.mean(test_loss)
			# TODO: add test loss to logger
			# TODO: get accuracy
			print('  [*] Test loss: %.4f'%test_loss)

			if test_loss < old_loss or (iteration % 1000 == 0):
				# always save at some iteration
				print("  [*] Save ckpt, iter: %d at ckpt/"%iteration)
				file_path = 'ckpt/caps_mnist_%d.pth'%(iteration)
				torch.save(net.state_dict(), file_path)
				if test_loss < old_loss:
					old_loss = test_loss

			# back to train mode
			net_train()
	torch.save(net.state_dict(), 'ckppt/caps_mnist_last.pth')

if __name__ == '__main__':
	train()
