import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as functional, torch.utils.data as torchdata
from dataloaders import dataloaders
from utils import *
from torchsummary import summary
import time, math, numpy as np, matplotlib.pyplot as plt, argparse, os, collections, sys, inspect, pprint, scipy.stats as st
from functools import partial

def pretrained_ae_files(dataset, filters, ds = None, bn = None, pen = None, lmbda = None):
	if dataset in ['imagenet2012', 'imagenetdownloader']:
		name = '.pth'
	elif dataset in ['mnist', 'cifar10', 'cifar100', 'tinyimagenet']:
		name = '-' + ('ds' if ds else '') + ('bn' if bn else '') + str(filters) + ('pen' + str(pen) if pen else '') + ('lmbda' + str(lmbda) if pen else '') + '.pth'
	encoder_pre, decoder_pre = dataset + '-encoder', dataset + '-decoder'
	encoder_file = os.path.join(os.getcwd(), 'autoencoders', encoder_pre + '-weights', encoder_pre + name)
	decoder_file = os.path.join(os.getcwd(), 'autoencoders', decoder_pre + '-weights', decoder_pre + name)
	return encoder_file, decoder_file

def get_autoencoder(dataset, filters, learnencoder, datashape = None, valloader = None, mean = None, std = None):
	ds = dataset in ['mnist', 'cifar10', 'tinyimagenet'] 
	bn = dataset in ['cifar100', 'tinyimagenet'] 
	encoder, decoder = create_autoencoder(1 if dataset == 'mnist' else 3, filters, ds, bn, dataset in ['imagenet2012', 'imagenetdownloader'])
	if not learnencoder:
		encoder_file, decoder_file = pretrained_ae_files(dataset, filters, ds, bn, 0, 0)
		encoder.load_state_dict(torch.load(encoder_file))
		decoder.load_state_dict(torch.load(decoder_file))
		encoder.eval()
		decoder.eval()
		test_autoencoder(datashape, encoder, decoder, valloader, mean, std)
	return encoder, decoder

def test_autoencoder(datashape, encoder, decoder, testloader, mean, std):
	print('\n--- Testing autoencoder')
	print('-' * 64, 'encoder\n', encoder)
	summary(encoder, datashape[1:], device = 'cpu')
	print('-' * 64, 'decoder\n', decoder)
	criterion = nn.MSELoss()
	test_loss, idx_batch = 0, 4
	for i, (x, _) in enumerate(testloader):
		z = encoder(x)
		y = decoder(z)
		loss = criterion(y, x)
		test_loss += loss.item()
		if i == idx_batch:
			idx_images = np.random.choice(x.size()[0], 5, replace = False)
			x_ = x.cpu().detach().numpy().copy()[idx_images, :, :, :]
			y_ = y.cpu().detach().numpy().copy()[idx_images, :, :, :]
			show_autoencoder_images(x_, y_, mean, std, 'test-ae2.png')
			break
	test_loss /= (i + 1)
	print('--- Autoencoder test loss : {:.4f}'.format(test_loss))

class FirstResBlock(nn.Module):
	def __init__(self, nfilters, batchnorm = True, bias = False, timestep = 1):
		super(FirstResBlock, self).__init__()
		self.timestep = timestep
		self.batchnorm = batchnorm
		self.cv1 = nn.Conv2d(nfilters, nfilters, 3, 1, 1, bias = bias)
		if self.batchnorm:
			self.bn2 = nn.BatchNorm2d(nfilters)
		self.cv2 = nn.Conv2d(nfilters, nfilters, 3, 1, 1, bias = bias)
	def forward(self, x):
		z = self.cv1(x)
		z = functional.relu(self.bn2(z)) if self.batchnorm else functional.relu(x)
		z = self.cv2(z)
		return x + self.timestep * z, z 

class ResBlock(nn.Module):
	def __init__(self, nfilters, batchnorm = True, bias = False, timestep = 1):
		super(ResBlock, self).__init__()
		self.timestep = timestep
		self.batchnorm = batchnorm
		if self.batchnorm :
			self.bn1 = nn.BatchNorm2d(nfilters)
		self.cv1 = nn.Conv2d(nfilters, nfilters, 3, 1, 1, bias = bias)
		if self.batchnorm :
			self.bn2 = nn.BatchNorm2d(nfilters)
		self.cv2 = nn.Conv2d(nfilters, nfilters, 3, 1, 1, bias = bias)
	def forward(self, x):
		z = functional.relu(self.bn1(x)) if self.batchnorm else functional.relu(x)
		z = self.cv1(z) 
		z = functional.relu(self.bn2(z)) if self.batchnorm else functional.relu(x)
		z = self.cv2(z)
		return x + self.timestep * z, z

class ResNetStage(nn.Module):
	def __init__(self, nblocks, nfilters, first = False, batchnorm = True, bias = False, timestep = 1):
		super(ResNetStage, self).__init__()
		self.blocks = nn.ModuleList([FirstResBlock(nfilters, batchnorm, bias, timestep) if i == 0 and first else ResBlock(nfilters, batchnorm, bias, timestep) for i in range(nblocks)])
	def forward(self, x):
		rs = []
		for block in self.blocks :
			x, r = block(x)
			rs.append(r)
		return x, rs

class OneRepResNet(nn.Module):
	def __init__(self, datashape, nclasses, learnencoder, encoder, nblocks = 9, nfilters = 64, classifier_name = '3Lin', batchnorm = True, bias = False, timestep = 1):
		super(OneRepResNet, self).__init__()
		self.classifier_name = classifier_name
		self.encoder = encoder
		self.stage1 = ResNetStage(nblocks, nfilters, True, batchnorm, bias, timestep)
		with torch.no_grad():
			featureshape = list(self.encoder(torch.ones(*datashape)).shape)
		self.classifier = create_classifier(classifier_name, nclasses, featureshape, nfilters)
		if not learnencoder:
			for param in self.encoder.parameters():
				param.requires_grad = False
	def forward(self, x):
		x = self.encoder(x)
		x, rs = self.stage1(x)
		if self.classifier_name[-3:] == 'Lin':
			x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x, rs

class ResNet(nn.Module):
	def __init__(self, datashape, nclasses, learnencoder, encoder, nblocks = 18, nfilters = 64, batchnorm = True, bias = False, timestep = 1):
		super(ResNet, self).__init__()
		self.encoder = encoder
		self.stage1 = ResNetStage(nblocks, nfilters, True, batchnorm, bias, timestep)
		self.cv1 = nn.Conv2d(nfilters, 2 * nfilters, 1, 2, 0, bias = False)
		self.stage2 = ResNetStage(nblocks, 2 * nfilters, False, batchnorm, bias, timestep)
		self.cv2 = nn.Conv2d(2 * nfilters, 4 * nfilters, 1, 2, 0, bias = False)
		self.stage3 = ResNetStage(nblocks, 4 * nfilters, False, batchnorm, bias, timestep)
		self.bn = nn.BatchNorm2d(4 * nfilters, track_running_stats = True)
		self.avgpool = nn.AvgPool2d(8, 8)
		with torch.no_grad():
			featuresize = self.forward_conv(torch.zeros(*datashape))[0].view(-1).shape[0]
		self.classifier = nn.Linear(featuresize, nclasses)
		if not learnencoder:
			for param in self.encoder.parameters():
				param.requires_grad = False
	def forward_conv(self, x):
		rs = dict()
		x = self.encoder(x)
		x, rs[1] = self.stage1(x)
		x = self.cv1(x)
		x, rs[2] = self.stage2(x)
		x = self.cv2(x)
		x, rs[3] = self.stage3(x)
		x = functional.relu(self.bn(x), inplace = True)
		x = self.avgpool(x)
		return x, [r for i in range(1, 4) for r in rs[i]]
	def forward(self, x):
		x, rs = self.forward_conv(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x, rs

class AvgPoolResNet(nn.Module):
	def __init__(self, datashape, nclasses, learnencoder, encoder, nblocks = 10, nfilters = 64, batchnorm = True, bias = False, timestep = 1):
		super(AvgPoolResNet, self).__init__()
		self.encoder = encoder
		self.stage1 = ResNetStage(nblocks, nfilters, True, batchnorm, bias, timestep)
		self.avgpool1 = nn.AvgPool2d(2, 2)
		self.stage2 = ResNetStage(nblocks, nfilters, False, batchnorm, bias, timestep)
		self.avgpool2 = nn.AvgPool2d(2, 2)
		self.stage3 = ResNetStage(nblocks, nfilters, False, batchnorm, bias, timestep)
		self.avgpool3 = nn.AvgPool2d(2, 2)
		self.bn = nn.BatchNorm2d(nfilters, track_running_stats = True)
		self.avgpool4 = nn.AvgPool2d(4, 4)
		with torch.no_grad():
			featuresize = self.forward_conv(torch.zeros(*datashape))[0].view(-1).shape[0]
		self.classifier = nn.Linear(featuresize, nclasses)
		if not learnencoder:
			for param in self.encoder.parameters():
				param.requires_grad = False
	def forward_conv(self, x):
		rs = dict()
		x = self.encoder(x)
		x, rs[1] = self.stage1(x)
		x = self.avgpool1(x)
		x, rs[2] = self.stage2(x)
		x = self.avgpool2(x)
		x, rs[3] = self.stage3(x)
		x = self.avgpool3(x)
		x = functional.relu(self.bn(x), inplace = True)
		x = self.avgpool4(x)
		return x, [r for i in range(1, 4) for r in rs[i]]
	def forward(self, x):
		x, rs = self.forward_conv(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x, rs

class WideResNet(nn.Module):
	def __init__(self, datashape, nclasses, nblocks = 4, batchnorm = True, bias = False, timestep = 1):
		super(WideResNet, self).__init__()
		self.cv0 = nn.Conv2d(datashape[1], 160, 3, 1, 1, bias = False)
		self.stage1 = ResNetStage(nblocks, 160, True, batchnorm, bias, timestep)
		self.cv1 = nn.Conv2d(160, 320, 1, 2, 0, bias = False)
		self.stage2 = ResNetStage(nblocks, 320, False, batchnorm, bias, timestep)
		self.cv2 = nn.Conv2d(320, 640, 1, 2, 0, bias = False)
		self.stage3 = ResNetStage(nblocks, 640, False, batchnorm, bias, timestep)
		self.bn = nn.BatchNorm2d(640, track_running_stats = True)
		self.avgpool = nn.AvgPool2d(8, 8)
		with torch.no_grad():
			featuresize = self.forward_conv(torch.zeros(*datashape))[0].view(-1).shape[0]
		self.classifier = nn.Linear(featuresize, nclasses)
	def forward_conv(self, x):
		rs = dict()
		x = self.cv0(x)
		x, rs[1] = self.stage1(x)
		x = self.cv1(x)
		x, rs[2] = self.stage2(x)
		x = self.cv2(x)
		x, rs[3] = self.stage3(x)
		x = functional.relu(self.bn(x), inplace = True)
		x = self.avgpool(x)
		return x, [r for i in range(1, 4) for r in rs[i]]
	def forward(self, x):
		x, rs = self.forward_conv(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x, rs

class ResNextBlock(nn.Module):
	def __init__(self, infilters = 256, planes = 64, expansion = 4, cardinality = 32, width = 4, base = 64, stride = 1, relu = True, residu = True, downsample = None):
		super(ResNextBlock, self).__init__()
		self.relu = relu 
		self.residu = residu
		self.intfilters = cardinality * math.floor(planes * width / base)
		self.outfilters = planes * expansion
		self.cv1 = nn.Conv2d(infilters, self.intfilters, 1, 1, 0, bias = False)
		self.bn1 = nn.BatchNorm2d(self.intfilters)
		self.cv2 = nn.Conv2d(self.intfilters, self.intfilters, 3, stride, 1, groups = cardinality, bias = False)
		self.bn2 = nn.BatchNorm2d(self.intfilters)
		self.cv3 = nn.Conv2d(self.intfilters, self.outfilters, 1, 1, 0, bias = False)
		self.bn3 = nn.BatchNorm2d(self.outfilters)
		self.downsample = downsample
	def forward(self, x):
		r = functional.relu(self.bn1(self.cv1(x)), inplace = True)
		r = functional.relu(self.bn2(self.cv2(r)), inplace = True)
		r = functional.relu(self.bn3(self.cv3(r)), inplace = True)
		if self.downsample is not None:
			x = self.downsample(x)
		if self.relu :
			z = functional.relu(x + r, inplace = True)
			if self.residu :
				r = z - x
		else :
			z = x + r 
		return z, r

class ResNextStage(nn.Module):
	def __init__(self, nb, inf = 256, pln = 64, exp = 4, card = 32, width = 4, base = 64, stride = 1, rel = True, res = True):
		super(ResNextStage, self).__init__()
		intf = pln * exp
		ds = nn.Sequential(nn.Conv2d(inf, intf, 1, stride, bias = False), nn.BatchNorm2d(intf)) if stride != 1 or inf != intf else None
		block = lambda i : ResNextBlock(inf, pln, exp, card, width, base, stride, rel, res, ds) if i == 0 else ResNextBlock(intf, pln, exp, card, width, base, 1, rel, res)
		self.blocks = nn.ModuleList([block(i) for i in range(nb)])
	def forward(self, x):
		rs = []
		for block in self.blocks :
			x, r = block(x)
			rs.append(r)
		return x, rs

class ResNext29(nn.Module):
	def __init__(self, datashape, nclasses, learnencoder, encoder, nblocks = [3, 3, 3], infilters = 64, planes = 64, expansion = 4, 
				 cardinality = 16, width = 64, base = 64, relu = True, residu = True):
		super(ResNext29, self).__init__()
		self.encoder = encoder
		self.stage1 = ResNextStage(nblocks[0], infilters * 1, planes * 1, expansion, cardinality, width, base, 1, relu, residu)
		self.stage2 = ResNextStage(nblocks[1], infilters * 4, planes * 2, expansion, cardinality, width, base, 2, relu, residu)
		self.stage3 = ResNextStage(nblocks[2], infilters * 8, planes * 4, expansion, cardinality, width, base, 2, relu, residu)
		self.avgpool = nn.AvgPool2d(7, 1)
		with torch.no_grad():
			featuresize = self.forward_conv(torch.zeros(*datashape))[0].view(-1).shape[0]
		self.classifier = nn.Linear(featuresize, nclasses)
		if not learnencoder:
			for param in self.encoder.parameters():
				param.requires_grad = False
	def forward_conv(self, x):
		rs = dict()
		x = self.encoder(x)
		x, rs[1] = self.stage1(x)
		x, rs[2] = self.stage2(x)
		x, rs[3] = self.stage3(x)
		x = self.avgpool(x)
		return x, [r for i in range(1, 4) for r in rs[i]]
	def forward(self, x):
		x, rs = self.forward_conv(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x, rs

class ResNext50(nn.Module):
	def __init__(self, datashape, nclasses, learnencoder, encoder, nblocks = [3, 4, 6, 3], infilters = 64, planes = 64, expansion = 4, 
				 cardinality = 32, width = 4, base = 64, relu = True, residu = True):
		super(ResNext50, self).__init__()
		self.encoder = encoder
		self.stage1 = ResNextStage(nblocks[0], infilters * 1, planes * 1, expansion, cardinality, width, base, 1, relu, residu)
		self.stage2 = ResNextStage(nblocks[1], infilters * 4, planes * 2, expansion, cardinality, width, base, 2, relu, residu)
		self.stage3 = ResNextStage(nblocks[2], infilters * 8, planes * 4, expansion, cardinality, width, base, 2, relu, residu)
		self.stage4 = ResNextStage(nblocks[3], infilters * 16, planes * 8, expansion, cardinality, width, base, 2, relu, residu)
		self.avgpool = nn.AvgPool2d(7 if datashape[-1] == 224 else 4, 1)
		with torch.no_grad():
			featuresize = self.forward_conv(torch.zeros(*datashape))[0].view(-1).shape[0]
		self.classifier = nn.Linear(featuresize, nclasses)
		if not learnencoder:
			for param in self.encoder.parameters():
				param.requires_grad = False
	def forward_conv(self, x):
		rs = dict()
		x = self.encoder(x)
		x, rs[1] = self.stage1(x)
		x, rs[2] = self.stage2(x)
		x, rs[3] = self.stage3(x)
		x, rs[4] = self.stage4(x)
		x = self.avgpool(x)
		return x, [r for i in range(1, 5) for r in rs[i]]
	def forward(self, x):
		x, rs = self.forward_conv(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x, rs

def getmodel(datashape, model, nclasses, learnencoder, encoder, nfilters, batchnorm, bias, timestep, classifier, nblocks, relu, residu):
	if model == 'resnext29':
		return ResNext29(datashape, nclasses, learnencoder, encoder, infilters = nfilters, relu = relu, residu = residu)
	if model == 'resnext50':
		return ResNext50(datashape, nclasses, learnencoder, encoder, infilters = nfilters, relu = relu, residu = residu,)
	if model == 'onerep':
		return OneRepResNet(datashape, nclasses, learnencoder, encoder, nblocks, nfilters, classifier, batchnorm, bias, timestep)
	if model == 'resnet':
		return ResNet(datashape, nclasses, learnencoder, encoder, nblocks, nfilters, batchnorm, bias, timestep)
	if model == 'avgpool':
		return AvgPoolResNet(datashape, nclasses, learnencoder, encoder, nblocks, nfilters, batchnorm, bias, timestep)
	if model == 'wide':
		return WideResNet(datashape, nclasses, nblocks, batchnorm, bias, timestep)
	else:
		raise NotImplementedError()

def train(model, optimizer, scheduler, criterion, trainloader, valloader, testloader, pnorm, lmt0, lml0, tau, uzs, nepochs, clip = 0):
	train_loss, train_acc1, train_acc5, val_loss, val_acc1, val_acc5, lmt, lml, t0, it, uzawa = [], [], [], [], [], [], lmt0, lml0, time.time(), 0, uzs > 0 and tau > 0
	lmt = 1 / lml if uzawa else lmt
	print('\n--- Begin trainning\n')
	for e in range(nepochs):
		model.train()
		t1, loss_meter, acc1_meter, acc5_meter, time_meter = time.time(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
		for j, (x, y) in enumerate(trainloader):
			t2, it = time.time(), it + 1
			x, y = x.to(device), y.to(device)
			optimizer.zero_grad()
			out, rs = model(x)
			classification_loss = criterion(out, y) 
			if uzawa and it % uzs == 0 :
				lml += tau * classification_loss.item()
				lmt = 1 / lml
			if lmt > 0 : 
				transport = sum([torch.mean(torch.abs(r) ** pnorm) for r in rs])
				loss = classification_loss + lmt * transport
			else :
				loss = classification_loss
			loss.backward()
			if clip > 0:
				torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
			optimizer.step()
			num = len(y)
			prec1, prec5 = topkaccuracy(out.data, y.data, topk = (1, 5))
			loss_meter.update(loss.item(), num)
			acc1_meter.update(prec1, num)
			acc5_meter.update(prec5, num)
			time_meter.update(time.time() - t2, 1)
			if j % 500 == 0 :
				m = (e + 1, nepochs, j + 1, len(trainloader), loss_meter.avg, acc1_meter.avg, acc5_meter.avg, lml, time_meter.avg)
				print('[Ep {:^5}/{:^5} Batch {:^5}/{:^5}] Train loss {:9.4f} Train top1acc {:.4f} Train top5acc {:.4f} Lambda loss {:9.4f} Batch time {:.4f}s'.format(*m))
		train_loss.append(loss_meter.avg)
		train_acc1.append(acc1_meter.avg)
		train_acc5.append(acc5_meter.avg)
		optimizer.zero_grad()
		vlo, vac1, vac5 = test(model, criterion, valloader)
		val_loss.append(vlo)
		val_acc1.append(vac1)
		val_acc5.append(vac5)
		m = (e + 1, nepochs, vlo, vac1, vac5, time.time() - t1, time.time() - t0)
		print('\n[***** Ep {:^5}/{:^5} over ******] Valid loss {:9.4f} Valid top1acc {:.4f} Valid top5acc {:.4f} Epoch time {:9.4f}s Total time {:.4f}s\n'.format(*m))
		scheduler.step()
	test_loss, test_acc1, test_acc5 = test(model, criterion, testloader)
	return train_loss, val_acc1, val_acc5

def test(model, criterion, loader):
	model.eval()
	loss_meter, acc1_meter, acc5_meter = AverageMeter(), AverageMeter(), AverageMeter()
	for j, (x, y) in enumerate(loader):
		with torch.no_grad():
			x, y = x.to(device), y.to(device)
			out, residus = model(x)
			ent = criterion(out, y)
			trs = sum([sum([torch.mean(r ** 2) for r in residus[i]]) for i in range(1, 5)])
			num = len(y)
			prec1, prec5 = topkaccuracy(out.data, y.data, topk = (1, 5))
			loss_meter.update(ent.item(), num)
			acc1_meter.update(prec1, num)
			acc5_meter.update(prec5, num)
	return loss_meter.avg, acc1_meter.avg, acc5_meter.avg
	

def experiment(dataset, modelname, learnencoder, pnorm, nfilters, learningrate, lambdatransport, lambdaloss0, tau, uzawasteps, batchnorm, bias, timestep, clip, classifier,
			   nblocks, nepochs, init, initname, initgain, trainsize, valsize, testsize, batchsize, relu, residu, weightdecay, seed = None, experiments = False):

	t0 = time.time()
	if seed is not None:
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		torch.manual_seed(seed)
		np.random.seed(seed)

	trainloader, valloader, testloader, datashape, nclasses, mean, std = dataloaders(dataset, batchsize, trainsize, valsize, testsize)

	if nepochs > 5:
		folder = modelname + '-' + dataset + '-' + time.strftime("%Y%m%d-%H%M%S") + '-lt' + str(lambdatransport) + '-ll' + str(lambdaloss0) + '-ta' + str(tau)
		make_folder(folder)
		stdout0 = sys.stdout
		sys.stdout = open(os.path.join(folder, 'log.txt'), 'wt')
	frame = inspect.currentframe()
	names, _, _, values = inspect.getargvalues(frame)
	print('experiment from main.py with parameters')
	for name in names:
		print('%s = %s' % (name, values[name]))
	if uzawasteps == 0 and (lambdaloss0 != 1 or tau > 0):
		print('us = 0 means no transport loss. lambda loss is fixed to 1, tau to 0, and lambda transport to 0')
		lambdaloss0, tau, lambdatransport = 1, 0, 0
	if uzawasteps > 0 and lambdatransport != 1:
		print('us > 0 means uzawa. lambda transport is fixed to 1')
		lambdatransport = 1
	print('train batches =', len(trainloader), 'val batches =', len(valloader))

	encoder, decoder = get_autoencoder(dataset, nfilters, learnencoder, datashape, valloader, mean, std)
	model = getmodel(datashape, modelname, nclasses, learnencoder, encoder, nfilters, batchnorm, bias, timestep, classifier, nblocks, relu, residu)
	if torch.cuda.device_count() > 1:
  		print('\n---', torch.cuda.device_count(), 'GPUs \n')
  		model = nn.DataParallel(model)
	initialization = partial(initialize, initname, initgain)
	if init:
		for name, module in model.named_modules():
			if name != 'module' and len(name) > 0 and name.count('.') == 1 and (learnencoder or 'encoder' not in name):
				module.apply(initialization) 
	model.to(device)
	print(model)
	
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr = learningrate, momentum = 0.9, weight_decay = weightdecay)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [150, 225, 250], gamma = 0.1)

	train_loss, val_acc1, val_acc5 = train(model, optimizer, scheduler, criterion, trainloader, valloader, testloader, pnorm, 
										   lambdatransport, lambdaloss0, tau, uzawasteps, nepochs)
	
	if experiments and nepochs > 5:
		print('--- Train Loss \n', train_loss, '\n--- Val Acc1 \n', val_acc1, '\n--- Val Acc5 \n', val_acc5)
		print('--- Min Train Loss \n', min(train_loss), '\n--- Max Val Acc1 \n', max(val_acc1), '\n--- Max Val Acc5 \n', max(val_acc5))
		sys.stdout.close()
		sys.stdout = stdout0

	if not experiments:
		torch.save(model.state_dict(), os.path.join(folder, 'weights.pth'))
	del model
	return train_loss, val_acc1, val_acc5, time.time() - t0

def experiments(parameters, average):
	t0, j, f = time.time(), 0, 110
	sep = '-' * f 
	acc1s, acc5s = [], []
	nparameters = len(parameters)
	nexperiments = int(np.prod([len(parameters[i][1]) for i in range(nparameters)]))
	print('\n' + sep, 'main.py')
	print(sep, nexperiments, 'experiments ' + ('to average ' if average else '') + 'over parameters:')
	pprint.pprint(parameters, width = f, compact = True)
	for params in product([values for name, values in parameters]) :
		j += 1
		print('\n' + sep, 'experiment %d/%d with parameters:' % (j, nexperiments))
		pprint.pprint([parameters[i][0] + ' = ' + str(params[i]) for i in range(nparameters)], width = f, compact = True)
		tr_loss, vl_acc1, vl_acc5, t1 = experiment(*params, True)
		acc1s.append(np.max(vl_acc1))
		acc5s.append(np.max(vl_acc5))
		print(sep, 'experiment %d/%d over. took %.1f s. total %.1f s' % (j, nexperiments, t1, time.time() - t0))
	acc1 = np.mean(acc1s)
	acc5 = np.mean(acc5s)

	confint1 = st.t.interval(0.95, len(acc1s) - 1, loc = acc1, scale = st.sem(acc1s))
	print('\nall test acc1', acc1s)
	print('\naverage test acc1', acc1)
	print('\nconfint1', confint1)

	confint5 = st.t.interval(0.95, len(acc5s) - 1, loc = acc5, scale = st.sem(acc5s))
	print('\nall test acc5', acc5s)
	print('\naverage test acc5', acc5)
	print('\nconfint5', confint5)
	
	print(('\n' if not average else '') + sep, 'total time for %d experiments: %.1f s' % (j, time.time() - t0))

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("-dat", "--dataset", required = True, choices = ['mnist', 'cifar10', 'cifar100', 'imagenet2012', 'tinyimagenet', 'imagenetdownloader'], nargs = '*')
	parser.add_argument("-mod", "--modelname", required = True, choices = ['resnext29', 'resnext50', 'onerep', 'resnet', 'avgpool', 'wide'], nargs = '*')
	parser.add_argument("-lec", "--learnencoder", type = int, default = [1], choices = [0, 1], nargs = '*')
	parser.add_argument("-pno", "--pnorm", type = int, default = [2], nargs = '*')
	parser.add_argument("-nfl", "--nfilters", type = int, default = [64], nargs = '*')
	parser.add_argument("-lrr", "--learningrate", type = float, default = [0.1], nargs = '*')
	parser.add_argument("-lmt", "--lambdatransport", type = float, default = [0], nargs = '*')
	parser.add_argument("-lml", "--lambdaloss0", type = float, default = [1], nargs = '*')
	parser.add_argument("-tau", "--tau", type = float, default = [0], nargs = '*')
	parser.add_argument("-uzs", "--uzawasteps", type = int, default = [0], nargs = '*')
	parser.add_argument("-btn", "--batchnorm", type = int, default = [1], choices = [0, 1], nargs = '*')
	parser.add_argument("-bia", "--bias", type = int, default = [0], choices = [0, 1], nargs = '*')
	parser.add_argument("-tst", "--timestep", type = float, default = [1], nargs = '*')
	parser.add_argument("-clp", "--clip", type = float, default = [0], nargs = '*')
	parser.add_argument("-cla", "--classifier", default = ['3Lin'], choices = ['1Lin', '2Lin', '3Lin'], nargs = '*')
	parser.add_argument("-nbl", "--nblocks", type = int, default = [9], nargs = '*')
	parser.add_argument("-nep", "--nepochs", type = int, default = [300], nargs = '*')
	parser.add_argument("-ini", "--init", type = int, default = [1], choices = [0, 1], nargs = '*')
	parser.add_argument("-inn", "--initname", default = ['kaiming'], choices = ['orthogonal', 'normal', 'kaiming'], nargs = '*')
	parser.add_argument("-ing", "--initgain", type = float, default = [0.01], nargs = '*')
	parser.add_argument("-trs", "--trainsize", type = float, default = [None], nargs = '*')
	parser.add_argument("-vls", "--valsize", type = float, default = [None], nargs = '*')
	parser.add_argument("-tss", "--testsize", type = float, default = [None], nargs = '*')
	parser.add_argument("-bas", "--batchsize", type = int, default = [128], nargs = '*')
	parser.add_argument("-rel", "--relu", type = int, default = [1], choices = [0, 1], nargs = '*')
	parser.add_argument("-res", "--residu", type = int, default = [1], choices = [0, 1], nargs = '*')
	parser.add_argument("-wdc", "--weightdecay", type = float, default = [0.0001], nargs = '*')
	parser.add_argument("-see", "--seed", type = int, default = [None], nargs = '*')
	parser.add_argument("-exp", "--experiments", action = 'store_true')
	parser.add_argument("-avg", "--averageexperiments", action = 'store_true')
	args = parser.parse_args()

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	if args.experiments or args.averageexperiments:
		parameters = [(name, values) for name, values in vars(args).items() if name not in ['experiments', 'averageexperiments']]
		experiments(parameters, args.averageexperiments)
	else :
		parameters = [values[0] for name, values in vars(args).items() if name not in ['experiments', 'averageexperiments']]
		experiment(*parameters, False)

	



