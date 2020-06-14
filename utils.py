import torch.nn as nn, os, numpy as np
import matplotlib as mpl, matplotlib.pyplot as plt, matplotlib.cm as cm
from scipy.spatial import ConvexHull
from sklearn.mixture import GaussianMixture
from scipy import linalg
from sklearn.neighbors import NearestNeighbors
from functools import partial
import torch.nn.functional as functional
from collections import OrderedDict

stack = lambda d :  {name: np.vstack(inp) for name, inp in d.items()}
get_avg = lambda d, n : [d[i].avg for i in range(n)]
l2norm = lambda x : np.sqrt(np.sum(x ** 2, axis = (1, 2, 3)))

convDiag = lambda x, M : functional.conv2d(x, M, stride = 1, padding = 1, groups = M.shape[0])
convDiagT = lambda x, M : functional.conv_transpose2d(x, M, stride = 1, padding = 1, groups = M.shape[0])

class ResnetClassifier(nn.Module):
	def __init__(self, feature_shape, nclasses, learn_bn = True, filters = None):
		super(ResnetClassifier, self).__init__()
		self.bn = nn.BatchNorm2d(filters) if learn_bn else nn.BatchNorm2d(feature_shape[1], affine = False, track_running_stats = False)
		self.avgpool = nn.AvgPool2d(avgpool_resnet_classifier, avgpool_resnet_classifier)
		self.feature_size = self.subsample(torch.zeros(*feature_shape)).view(-1).shape[0]
		self.fc = nn.Linear(self.feature_size, nclasses)
	def subsample(self, x):
		x = self.avgpool(functional.relu(self.bn(x)))
		return x
	def forward(self, x):
		x = self.subsample(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x

def topkaccuracy(output, target, topk = (1, )):
	maxk = max(topk)
	num = len(target)
	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))
	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.item() / num)
	return res

def create_autoencoder(in_channels = 3, filters = 100, ds = True, bn = True, imagenet = False):
	int_filters = int(filters / 2)
	if imagenet:
		encoder = nn.Sequential(nn.Conv2d(3, filters, 7, 2, 3), nn.BatchNorm2d(filters), nn.ReLU(True), 
								nn.Conv2d(filters, filters, 3, 2, 1), nn.BatchNorm2d(filters), nn.ReLU(True))
		decoder = nn.Sequential(nn.ConvTranspose2d(filters, filters, 3, 2, 1, 1), nn.BatchNorm2d(filters), nn.ReLU(True), 
								nn.ConvTranspose2d(filters, 3, 7, 2, 3, 1), nn.BatchNorm2d(3), nn.Tanh())
		return encoder, decoder
	if not ds:
		if not bn:
			encoder = nn.Sequential(nn.Conv2d(in_channels, int_filters, 3, 1, 1), nn.ReLU(True),
            						nn.Conv2d(int_filters, filters, 3, 1, 1), nn.ReLU(True))
			decoder = nn.Sequential(nn.Conv2d(filters, int_filters, 3, 1, 1), nn.ReLU(True),
            						nn.Conv2d(int_filters, in_channels, 3, 1, 1), nn.Tanh())
		else:
			encoder = nn.Sequential(nn.Conv2d(in_channels, int_filters, 3, 1, 1), nn.BatchNorm2d(int_filters), nn.ReLU(True),
            						nn.Conv2d(int_filters, filters, 3, 1, 1), nn.BatchNorm2d(filters), nn.ReLU(True))
			decoder = nn.Sequential(nn.Conv2d(filters, int_filters, 3, 1, 1), nn.BatchNorm2d(int_filters), nn.ReLU(True),
            						nn.Conv2d(int_filters, in_channels, 3, 1, 1), nn.BatchNorm2d(in_channels), nn.Tanh())
	else:
		if not bn:	
			encoder = nn.Sequential(nn.Conv2d(in_channels, int_filters, 3, 1, 1), nn.ReLU(True),
       	    						nn.Conv2d(int_filters, filters, 5, 2, 2), nn.ReLU(True))
			decoder = nn.Sequential(nn.ConvTranspose2d(filters, int_filters, 5, 2, 2, 1), nn.ReLU(True),
            						nn.ConvTranspose2d(int_filters, in_channels, 3, 1, 1), nn.Tanh())
		else:
			encoder = nn.Sequential(nn.Conv2d(in_channels, int_filters, 3, 1, 1), nn.BatchNorm2d(int_filters), nn.ReLU(True),
       	    						nn.Conv2d(int_filters, filters, 5, 2, 2), nn.BatchNorm2d(filters), nn.ReLU(True))
			decoder = nn.Sequential(nn.ConvTranspose2d(filters, int_filters, 5, 2, 2, 1), nn.BatchNorm2d(int_filters), nn.ReLU(True),
            						nn.ConvTranspose2d(int_filters, in_channels, 3, 1, 1), nn.BatchNorm2d(in_channels), nn.Tanh())
	return encoder, decoder

	

def create_classifier(name, nclasses, feature_shape, filters = None):
	feature_size = np.prod(feature_shape)
	if name == '1Lin':
		return nn.Linear(feature_size, nclasses)
	if name == '2Lin':
		return nn.Sequential(nn.Linear(feature_size, nclasses * 10), nn.Sigmoid(), nn.Linear(nclasses * 10, nclasses))
	if name == '3Lin':
		return nn.Sequential(nn.Linear(feature_size, nclasses * 10), nn.BatchNorm1d(nclasses * 10), nn.ReLU(True), nn.Linear(nclasses * 10, nclasses))
	if name == 'ResNet':
		return ResnetClassifier(feature_shape, nclasses, True, filters)
	if name == 'ResNetNoBN':
		return ResnetClassifier(feature_shape, nclasses, False)

def initialize(name, gain, module):
	if name == 'orthogonal':
		init = partial(nn.init.orthogonal_, gain = gain) 
	elif name == 'normal':
		init = partial(nn.init.normal_, mean = 0, std = gain) 
	elif name == 'kaiming':
		init = partial(nn.init.kaiming_normal_, a = 0, mode = 'fan_out', nonlinearity = 'relu')
	else:
		raise ValueError('Unknown init ' + name)
	if isinstance(module, nn.Conv2d):
		init(module.weight)
		if hasattr(module, 'bias') and module.bias is not None:
			nn.init.constant_(module.bias, 0)
	elif isinstance(module, nn.BatchNorm2d):
		if hasattr(module, 'weight') and module.weight is not None:
			nn.init.constant_(module.weight, 1)
		if hasattr(module, 'bias') and module.bias is not None:
			nn.init.constant_(module.bias, 0)
	elif isinstance(module, nn.Linear):
		init(module.weight)
		if hasattr(module, 'bias') and module.bias is not None:
			nn.init.constant_(module.bias, 0)

def product(iterables):
    if len(iterables) == 0 :
        yield ()
    else :
        it = iterables[0]
        for item in it :
            for items in product(iterables[1: ]) :
                yield (item, ) + items

def make_folder(folder):
	if not os.path.exists(folder):
		os.makedirs(folder)

class AverageMeter(object):
	def __init__(self):
		self.reset()
	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0
	def update(self, val, num):
		self.val = val
		self.sum += val * num
		self.count += num
		self.avg = self.sum / self.count

def update_meters(y, pred, loss, loss_meter, acc_meter, trs = None, trs_meter = None, t = None, time_meter = None):
	num = len(y)
	correct = (pred == y).sum().item()
	accuracy = correct / num
	loss_meter.update(loss, num)
	acc_meter.update(accuracy, num)
	if trs is not None and trs_meter is not None:
		trs_meter.update(trs, num)
	if t is not None and time_meter is not None :
		time_meter.update(t, 1)

def plotscores(losses, accuracy, name):
	plt.figure(1)
	plt.subplot(211)
	plt.plot(losses)
	plt.ylabel('train loss')
	plt.subplot(212)
	plt.plot(accuracy)
	plt.xlabel('epoch')
	plt.ylabel('test accuracy')
	plt.savefig(name + '-loss-acc.png', bbox_inches = 'tight')
	plt.close()

def show_autoencoder_images(x, y, mean, std, name = None):
	bw = x.shape[1] == 1
	x = x[:, 0, :, :] if bw else np.moveaxis(x, 1, -1)
	y = y[:, 0, :, :] if bw else np.moveaxis(y, 1, -1)
	r, c = 2, x.shape[0]
	fig, axs = plt.subplots(r, c)
	for j in range(c):
		img = x[j, :, :] if bw else x[j, :, :, :]
		img = np.clip(std * img + mean, 0, 1)
		axs[0, j].imshow(img, cmap = 'gray' if bw else None)
		axs[0, j].axis('off')
		img = y[j, :, :] if bw else y[j, :, :, :]
		img = np.clip(std * img + mean, 0, 1)
		axs[1, j].imshow(img, cmap = 'gray' if bw else None)
		axs[1, j].axis('off')
	if name is not None:
		fig.savefig(name, bbox_inches = 'tight')
	else:
		plt.show()
	plt.close()

def show_decoded_images(images, mean, std, name = None):
	bw = images[0].shape[1] == 1
	rows, cols = images[0].shape[0], len(images)
	images = [img[:, 0, :, :] for img in images] if bw else [np.moveaxis(img, 1, -1) for img in images]
	col_names = ['og', 'ae'] + ['b{}'.format(i + 1) for i in range(cols - 2)]
	fig, axs = plt.subplots(rows, cols)
	for r in range(rows):
		for c in range(cols):
			img = images[c][r, :, :] if bw else images[c][r, :, :, :]
			img = np.clip(std * img + mean, 0, 1)
			axs[r, c].imshow(img, cmap = 'gray' if bw else None)
			axs[r, c].set_xticks([])
			axs[r, c].set_yticks([])
	for ax, col_name in zip(axs[0], col_names):
		ax.set_title(col_name)
	if name is not None:
		fig.savefig(name, bbox_inches = 'tight')
	else:
		plt.show()
	plt.close()

def plot_arrays(ratios, cosines, forcings, distances, nblocks, epoch, folder):
	plt.figure(figsize = (7, 7)) # plot cosine loss
	plt.plot(list(range(1, nblocks + 1)), cosines)
	plt.title('cos(F, grad L) after epoch ' + str(epoch))
	plt.xlabel('block')
	plt.ylabel('cos( F(h), grad_h L )')
	plt.savefig(os.path.join(folder, 'cos_epoch' + str(epoch) + '.png'), bbox_inches = 'tight')
	plt.close()
	plt.figure(figsize = (7, 7)) # plot forcing function and W2 movement
	plt.plot(list(range(1, nblocks + 1)), forcings, 'b', label = 'F(x)')
	if distances is not None:
		plt.plot(list(range(1, nblocks + 1)), distances, 'r', label = 'W2 movement')
	plt.title(('F and wasserstein distance' if distances is not None else 'F') + ' after epoch ' + str(epoch))
	plt.xlabel('block')
	plt.legend(loc = 'best')
	plt.savefig(os.path.join(folder, 'distance_epoch' + str(epoch) + '.png'), bbox_inches = 'tight')
	plt.close()
	plt.figure(figsize = (7, 7)) # plot cosine loss
	plt.plot(list(range(1, nblocks + 1)), ratios)
	plt.title('forcing function to input norm ratio after epoch ' + str(epoch))
	plt.xlabel('block')
	plt.ylabel('F(x) / x')
	plt.savefig(os.path.join(folder, 'ratio_epoch' + str(epoch) + '.png'), bbox_inches = 'tight')
	plt.close()

def convexHulls(points, labels):    
	convex_hulls = []
	for i in range(10):
		convex_hulls.append(ConvexHull(points[labels==i,:]))    
	return convex_hulls
    
def best_ellipses(points, labels):  
    gaussians = []    
    for i in range(10):
        gaussians.append(GaussianMixture(n_components=1, covariance_type='full').fit(points[labels==i, :])) 
    return gaussians
    
def Visualization(points2D, labels, convex_hulls, ellipses , nh, name = None, projname = 'tSNE'): 
    points2D_c = []
    for i in range(10):
        points2D_c.append(points2D[labels == i, :])
    cmap = cm.tab10 
    
    plt.figure(figsize=(3.841, 7.195), dpi = 100)
    plt.set_cmap(cmap)
    plt.subplots_adjust(hspace = 0.4)
    plt.subplot(311)
    plt.scatter(points2D[:, 0], points2D[:, 1], c = labels, s = 3, edgecolors = 'none', cmap = cmap, alpha = 1.0)# cmap=cm.Vega10cmap= , alpha=0.2)
    plt.colorbar(ticks = range(10))
    plt.title("2D " + projname + " - NH=" + str(nh*100.0))
    
    vals = [i / 10.0 for i in range(10)]
    sp2 = plt.subplot(312)
    for i in range(10):
        ch = np.append(convex_hulls[i].vertices,convex_hulls[i].vertices[0])
        sp2.plot(points2D_c[i][ch, 0], points2D_c[i][ch, 1], '-', label = '$%i$'%i, color = cmap(vals[i]))         
    plt.colorbar(ticks = range(10))
    plt.title(projname +" Convex Hulls")
  	
    def plot_results(X, Y_, means, covariances, index, title, color):
        splot = ax
        for i, (mean, covar) in enumerate(zip(means, covariances)):
            v, w = linalg.eigh(covar)
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / linalg.norm(w[0])
            
            if not np.any(Y_ == i):
                continue
            plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color, alpha = 0.2)
    
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi  
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
            ell.set_clip_box(splot.bbox)
            ell.set_alpha(0.6)
            splot.add_artist(ell)
    
        plt.title(title)
    
    ax = plt.subplot(3, 1, 3)
    for i in range(10):
        plot_results(points2D[labels==i, :], ellipses[i].predict(points2D[labels==i, :]), ellipses[i].means_, ellipses[i].covariances_, 0,projname+" fitting ellipses", cmap(vals[i]))
    if name is not None:
      plt.savefig(name, bbox_inches = 'tight', pi = 100)
    else:
      plt.show()
    plt.close()    
 
def neighboring_hit(points, labels):
	k = 6
	nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(points)
	distances, indices = nbrs.kneighbors(points)
	txs = 0.0
	txsc = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
	nppts = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
	for i in range(len(points)): 
		tx = 0.0
		for j in range(1,k+1):
			if labels[indices[i,j]]== labels[i]:
				tx += 1          
		tx /= k  
		txsc[labels[i]] += tx
		nppts[labels[i]] += 1
		txs += tx
	for i in range(10):
		txsc[i] /= nppts[i]
	return txs / len(points)






