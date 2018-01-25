from torch.utils.data import DataLoader
from torchvision import transforms, datasets


data_transform = transfroms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
		])

def load_mnist(n_worker = 4, batch_size = 64, split = 'train'):

	if train.lower() == 'train':
		trainset = datasets.MNIST('./data', train = True, download = True,
				transform = data_transform)
		train_loader = DataLoader(trainset, batch_size = batch_size,
				shuffle = True, pin_memory = True)
		return trainset, train_loader
	else:
		testset = datasets.MNIST('./data', train = False, download = True,
				transform = data_transform)
		train_loader = DataLoader(testset, batch_size = batch_size,
				shuffle = False, pin_memory = True)
		return testset, test_loader
