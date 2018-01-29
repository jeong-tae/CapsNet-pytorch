from visual import Logger

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
from torch.autograd import Variable

from models.CapsNet import capsNet
from models.Decoder import decoder
from models.Loss import marginLoss, reconstructionLoss
from data.DataLoader import load_mnist
from models.utils import accuracy

import numpy as np
import argparse, time

#TODO: take hyper-params as args and put into model config
parser = argparse.ArgumentParser(description = "Training arguments for CapsNet")
parser.add_argument('--max_iter', type = int, default = 100000, help = "Maximum training iterations")
parser.add_argument('--batch_size', type = int, default = 128, help = "Mini-batch size per iteration")
parser.add_argument('--lr', type = float, default = 0.01, help = "Learning rate to train")
parser.add_argument('--num_classes', type = int, default = 10, help = "Number of labels")
parser.add_argument('--r_iterations', type = int, default = 3, help = "Routing iteration")
parser.add_argument('--cuda', action = "store_true", default = False, help = "Use cuda to train")
parser.add_argument('--ckpt_postfix', type = str, default = 'MNIST', help = "Add postfix to check point name like caps_$postfix_$iteration.pth")
parser.add_argument('--log_dir', type = str, default = 'MNIST', help = "Directory for saving a logs, will be stored in visual/")

args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print(" [*] Set Cuda: True")
else:
    torch.set_default_tensor_type('torch.FloatTensor')
    print(" [*] Set Cuda: False")


logger = Logger('./visual/' + args.log_dir)

# TODO: weight load
net = capsNet(args) # first argument should be network configuration for building a network.
decoder_net = decoder(args)

if args.cuda:
    net = torch.nn.DataParallel(net)
    decoder_net = torch.nn.DataParallel(decoder_net)
    cudnn.benchmark = True

margin_loss = marginLoss()
recon_loss = reconstructionLoss()

parameters = set(net.parameters()) | set(decoder_net.parameters())

opt = optim.Adam(parameters, lr = args.lr)
print(" [*] Training is ready now!")
# TODO: set logger

def one_hot(label, n_cls):
    """
        pytorch doesn't support one-hot encoding for sparse labels
        We will use this till support one-hot ft

        This mess up the code... We can use tensorflow version of mnist

        Args
            label: [batch,], it contains index of label
            n_cls: scalar, Number of classes
        return
            y: [batch, n_cls], one-hot encoded vector
    """
    y = torch.zeros(label.size(0), n_cls)
    for i in range(label.size(0)):
        y[i, label[i]] = 1.0
    return y

def train():
    net.train()
    m_loss = 0
    r_loss = 0
    old_loss = 999. # initial loss to compare with current loss
    epoch = -1

    steps = 0

    print(" [*] Training on MNIST dataset")
    batch_iterator = None
    trainset, train_loader = load_mnist(n_worker = 4, batch_size = args.batch_size, split = 'train')
    testset, test_loader = load_mnist(n_worker = 4, batch_size = args.batch_size, split = 'test')

    epoch_size = len(trainset) // args.batch_size
    for iteration in range(0, args.max_iter):
        if (not batch_iterator) or (iteration % epoch_size == 0):
            # create
            batch_iterator = iter(train_loader)
        # TODO?: lr decay??

        if iteration % epoch_size == 0:
            epoch += 1

        images, targets = next(batch_iterator)
        targets = one_hot(targets, args.num_classes)

        images = Variable(images)
        targets = Variable(targets)
        if args.cuda:
            images = images.cuda()
            targets = targets.cuda()

        t0 = time.time()
        out = net(images)
        recon_images = decoder_net(out, targets)
        opt.zero_grad()
        m_loss, r_loss = margin_loss(out, targets), recon_loss(recon_images, images)
        loss = (m_loss + r_loss) # r_loss not decreasing now...
        loss.backward()
        opt.step()
        t1 = time.time()

        # TODO: logger...
        #================ TensorBoard logging ================#
        logger.scalar_summary('m_loss', m_loss.data[0], iteration + 1)
        logger.scalar_summary('r_loss', r_loss.data[0], iteration + 1)
        logger.scalar_summary('loss', m_loss.data[0] + r_loss.data[0], iteration + 1)


        if (iteration % 10) == 0: # Display period
            print(' [*] Epoch[%d], Iter %d || Loss: %.4f || m_loss: %.4f || r_loss: %.4f || Timer: %.4f sec'%(epoch, iteration, loss.data[0], m_loss.data[0], r_loss.data[0], (t1 - t0)))
            
            recon_images = recon_images.view(-1, 28, 28)[:10]
            recon_images = recon_images.data.cpu().numpy()
            logger.image_summary('reconstruction images', recon_images, iteration + 1)

        if (iteration % 200) == 0: # Eval period
            # set to net eval mode
            net.eval()

            test_loss = []
            total_acc = []
            for test_images, test_targets in test_loader:
                test_targets = one_hot(test_targets, args.num_classes)
                test_images = Variable(test_images)
                test_targets = Variable(test_targets)
                if args.cuda:
                    test_images = test_images.cuda()
                    test_targets = test_targets.cuda()

                out = net(test_images)
                m_loss = margin_loss(out, test_targets)
                # r_loss = recon_loss(out, test_images)
                loss = m_loss.data[0]
                # loss = (m_loss.data[0], r_loss.data[0])
                test_loss.append(loss)

                acc = accuracy(out, test_targets)
                total_acc.append(acc)

            test_loss = np.mean(test_loss)
            test_acc = np.mean(total_acc)
            # TODO: add test loss and acc to logger
            print('  [*] Test loss: %.4f, Test acc: %.4f'%(test_loss, test_acc))
            logger.scalar_summary('test_loss', test_loss, iteration + 1)
            logger.scalar_summary('test_acc', test_acc, iteration + 1)

            if test_loss < old_loss or (iteration % 1000 == 0):
                # always save at some iteration
                print("  [*] Save ckpt, iter: %d at ckpt/"%iteration)
                file_path = 'ckpt/caps_%s_%d.pth'%(args.ckpt_postfix, iteration)
                torch.save(net.state_dict(), file_path)
                if test_loss < old_loss:
                    old_loss = test_loss

            # TODO: get reconstruction samples

            # back to train mode
            # TODO: decoder also go back to train mode
            net.train()
    torch.save(net.state_dict(), 'ckppt/caps_%s_last.pth'%args.ckpt_postfix)

if __name__ == '__main__':
    train()
