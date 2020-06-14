import torch, torch.utils.data as torchdata
import torchvision, torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

def mean_and_std(datafolder = './train', batchsize = 300000, nworkers = 4):
	trainset = torchvision.datasets.ImageFolder(root = datafolder, transform = transforms.ToTensor())
	trainloader = torchdata.DataLoader(trainset, batch_size = batchsize, shuffle = False, num_workers = nworkers)
	pop_mean = []
	pop_std0 = []
	pop_std1 = []
	print(len(trainloader))
	for i, (x, y) in enumerate(trainloader):
		numpy_image = x.numpy()
		batch_mean = np.mean(numpy_image, axis = (0, 2 ,3))
		batch_std0 = np.std(numpy_image, axis = (0, 2 ,3))
		batch_std1 = np.std(numpy_image, axis = (0, 2, 3), ddof = 1)
		pop_mean.append(batch_mean)
		pop_std0.append(batch_std0)
		pop_std1.append(batch_std1)
	pop_mean = list(np.array(pop_mean).mean(axis = 0))
	pop_std0 = list(np.array(pop_std0).mean(axis = 0))
	pop_std1 = list(np.array(pop_std1).mean(axis = 0))	
	return pop_mean, pop_std0, pop_std1

def cifar10_dataloaders(batchsize, trainsize = 1, valsize = 0.5, testsize = 0.5):
	datashape, mean, std, nclasses = (1, 3, 32, 32), (0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784), 10
	transform = [transforms.ToTensor(), transforms.Normalize(mean, std)]
	data_aug = [transforms.RandomCrop(datashape[-1], padding = 4), transforms.RandomHorizontalFlip()]
	trainset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transforms.Compose(data_aug + transform))
	testset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = transforms.Compose(transform))
	trainloader = get_subset_loader(trainset, batchsize, trainsize)
	valloader, testloader = get_subset_loaders(testset, batchsize, [valsize, testsize])
	return trainloader, valloader, testloader, datashape, nclasses, np.array(mean), np.array(std)

def cifar100_dataloaders(batchsize, trainsize = 1, valsize = 0.5, testsize = 0.5):
	datashape, mean, std, nclasses = (1, 3, 32, 32), (0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784), 100
	transform = [transforms.ToTensor(), transforms.Normalize(mean, std)]
	data_aug = [transforms.RandomCrop(datashape[-1], padding = 4), transforms.RandomHorizontalFlip()]
	trainset = torchvision.datasets.CIFAR100(root = './data', train = True, download = True, transform = transforms.Compose(data_aug + transform))
	testset = torchvision.datasets.CIFAR100(root = './data', train = False, download = True, transform = transforms.Compose(transform))
	trainloader = get_subset_loader(trainset, batchsize, trainsize)
	valloader, testloader = get_subset_loaders(testset, batchsize, [valsize, testsize])
	return trainloader, valloader, testloader, datashape, nclasses, np.array(mean), np.array(std)

def mnist_dataloaders(batchsize, trainsize = 1, valsize = 0.5, testsize = 0.5):
	datashape, mean, std, nclasses = (1, 1, 28, 28), (0.1306604762738429, ), (0.30810780717887876, ), 10
	transform = [transforms.ToTensor(), transforms.Normalize(mean, std)]
	trainset = torchvision.datasets.MNIST(root = './data', train = True, download = True, transform = transforms.Compose(transform))
	testset = torchvision.datasets.MNIST(root = './data', train = False, download = True, transform = transforms.Compose(transform))
	trainloader = get_subset_loader(trainset, batchsize, trainsize)
	valloader, testloader = get_subset_loaders(testset, batchsize, [valsize, testsize])
	return trainloader, valloader, testloader, datashape, nclasses, np.array(mean), np.array(std)

def imagenet2012_dataloaders(batchsize, trainsize = 0.98, valsize = 0.01, testsize = 0.01):
	datashape, mean, std, nclasses = (1, 3, 224, 224), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 1000
	transform = [transforms.ToTensor(), transforms.Normalize(mean, std)]
	data_aug = [transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip()]
	test_trs = [transforms.Resize(256), transforms.CenterCrop(224)]
	trainset = torchvision.datasets.ImageFolder(root = './data/ILSVRC2012/ILSVRC2012_img_train', transform = transforms.Compose(data_aug + transform))
	trainloader, valloader, testloader = get_subset_loaders(trainset, batchsize, [trainsize, valsize, testsize])
	return trainloader, valloader, testloader, datashape, nclasses, np.array(mean), np.array(std)

def imagenetdownloader_dataloaders(batchsize, trainsize = 0.98, valsize = 0.01, testsize = 0.01):
	datashape, mean, std, nclasses = (1, 3, 224, 224), [0.47730196, 0.44212466, 0.38233677], [0.26841885, 0.2581342, 0.27384633], 493
	transform = [transforms.ToTensor(), transforms.Normalize(mean, std)]
	data_aug = [transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip()]
	test_trs = [transforms.Resize(256), transforms.CenterCrop(224)]
	trainset = torchvision.datasets.ImageFolder(root = './data/imagenet-downloader/imagenet-downloader-500-500-seed0', transform = transforms.Compose(data_aug + transform))
	trainloader, valloader, testloader = get_subset_loaders(trainset, batchsize, [trainsize, valsize, testsize])
	return trainloader, valloader, testloader, datashape, nclasses, np.array(mean), np.array(std)

def tinyimagenet_dataloaders(batchsize, trainsize = 1, valsize = 0.5, testsize = 0.5):
	datashape, mean, std, nclasses = (1, 3, 64, 64), [0.4802486, 0.44807222, 0.39754647], [0.2769859, 0.26906505, 0.2820814], 200
	transform = [transforms.ToTensor(), transforms.Normalize(mean, std)]
	data_aug = [transforms.RandomCrop(datashape[-1], padding = 4), transforms.RandomHorizontalFlip()]
	trainset = torchvision.datasets.ImageFolder(root = './data/tinyimagenet/train', transform = transforms.Compose(data_aug + transform))
	valset = torchvision.datasets.ImageFolder(root = './data/tinyimagenet/val', transform = transforms.Compose(transform))
	trainloader = get_subset_loader(trainset, batchsize, trainsize)
	valloader, testloader = get_subset_loaders(valset, batchsize, [valsize, testsize])
	return trainloader, valloader, testloader, datashape, nclasses, np.array(mean), np.array(std)

def get_subset_loader(dataset, batchsize, size):
	n = len(dataset)
	(sampler, shuffle) = (None, True) if size in [None, 'all', 0, 1] else (SubsetRandomSampler(np.random.choice(range(n), int(size * n), False)), False)
	return torchdata.DataLoader(dataset, batch_size = batchsize, shuffle = shuffle, num_workers = 2, sampler = sampler)

def get_subset_loaders(dataset, batchsize, sizes):
	if sum(sizes) > 1:
		raise ValueError('Sizes cannot sum to more than 1')
	n, s = len(dataset), len(sizes)
	indices = list(range(n))
	np.random.shuffle(indices)
	cutoffs = [0] + list(np.cumsum([int(np.floor(size * n)) for size in sizes]))
	idxs = [indices[cutoffs[i]: cutoffs[i + 1]] for i in range(s)]
	samplers = [SubsetRandomSampler(idx) for idx in idxs]
	loaders = [torchdata.DataLoader(dataset, batch_size = batchsize, shuffle = False, num_workers = 2, sampler = sampler) for sampler in samplers]
	return loaders

def fake_dataloaders(batchsize, datashape, nclasses):
	transform = [transforms.ToTensor()]
	trainset = torchvision.datasets.FakeData(size = 400, image_size = datashape[1:], num_classes = nclasses, transform = transforms.Compose(transform))
	trainloader = torchdata.DataLoader(trainset, batch_size = batchsize, shuffle = True, num_workers = 2)
	valset = torchvision.datasets.FakeData(size = 200, image_size = datashape[1:], num_classes = nclasses, transform = transforms.Compose(transform))
	valloader = torchdata.DataLoader(valset, batch_size = batchsize, shuffle = True, num_workers = 2)
	testset = torchvision.datasets.FakeData(size = 200, image_size = datashape[1:], num_classes = nclasses, transform = transforms.Compose(transform))
	testloader = torchdata.DataLoader(testset, batch_size = batchsize, shuffle = True, num_workers = 2)
	return trainloader, valloader, testloader, datashape, nclasses, None, None

def dataloaders(name, batchsize, trainsize = None, valsize = None, testsize = None):
	kwargs = {k : v for k, v in dict(batchsize = batchsize, trainsize = trainsize, valsize = valsize, testsize = testsize).items() if v is not None}
	if name == 'tinyimagenet':
		return tinyimagenet_dataloaders(**kwargs)
	if name == 'imagenet2012':
		return imagenet2012_dataloaders(**kwargs)
	if name == 'imagenetdownloader':
		return imagenetdownloader_dataloaders(**kwargs)
	if name == 'mnist':
		return mnist_dataloaders(**kwargs)
	if name == 'cifar10':
		return cifar10_dataloaders(**kwargs)
	if name == 'cifar100':
		return cifar100_dataloaders(**kwargs)
	if name == 'fake_like_mnist':
		return fake_dataloaders(batchsize, (1, 1, 28, 28), 10)
	if name == 'fake_like_cifar10':
		return fake_dataloaders(batchsize, (1, 3, 32, 32), 10)
	if name == 'fake_like_cifar100':
		return fake_dataloaders(batchsize, (1, 3, 32, 32), 100)
	else:
		raise ValueError('unknown dataset: ' + name)




