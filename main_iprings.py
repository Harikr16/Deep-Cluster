# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pdb
import argparse
import os
import pickle
import time

import faiss
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from skimage import transform

import clustering
import models
from util import AverageMeter, Logger, UnifLabelSampler
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--arch', '-a', type=str, metavar='ARCH',
					choices=['alexnet', 'vgg16','resnet','densenet'], default='alexnet',
					help='CNN architecture (default: alexnet)')
parser.add_argument('--sobel', action='store_true', help='Sobel filtering')
parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
					default='Kmeans', help='clustering algorithm (default: Kmeans)')
parser.add_argument('--nmb_cluster', '--k', type=int, default=10000,
					help='number of cluster for k-means (default: 10000)')
parser.add_argument('--lr', default=0.05, type=float,
					help='learning rate (default: 0.05)')
parser.add_argument('--wd', default=-5, type=float,
					help='weight decay pow (default: -5)')
parser.add_argument('--reassign', type=float, default=1.,
					help="""how many epochs of training between two consecutive
					reassignments of clusters (default: 1)""")
parser.add_argument('--workers', default=4, type=int,
					help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', type=int, default=200,
					help='number of total epochs to run (default: 200)')
parser.add_argument('--start_epoch', default=0, type=int,
					help='manual epoch number (useful on restarts) (default: 0)')
parser.add_argument('--batch', default=256, type=int,
					help='mini-batch size (default: 256)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
					help='path to checkpoint (default: None)')
parser.add_argument('--checkpoint', type=int, default=25000,
					help='how many iterations between two checkpoints (default: 25000)')
parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
parser.add_argument('--exp', type=str, default='', help='path to exp folder')
parser.add_argument('--verbose', action='store_true', help='chatty')
parser.add_argument('--num_classes', type=int, default=1000, help='No: of classes (default: 1000)')
parser.add_argument('--update_loader',type = int, default = 3, help = 'No: of epochs to improve resolution')


class Resize(object):
	def __init__(self,output_size = 512):
		self.output_size = output_size
		pass
	def __call__(self, image):
		# image , label = sample[0] , sample[1]
		# pdb.set_trace()
		h,w = image.size[:2]
		flag = 0
		if isinstance(self.output_size, int):
			if h < w:
				flag = 1
				new_h, new_w = self.output_size * h / w, self.output_size
			else:
				new_h, new_w = self.output_size, self.output_size * w / h
		else:
			new_h, new_w = self.output_size

		diff = abs(new_h -new_w)
		if diff%2 == 0:
			pad_l = diff/2
			pad_r = diff/2
		else:
			pad_l =diff//2
			pad_r = diff//2 +1
		# pad = abs((new_h - new_w)//2)
		new_h,new_w = int(new_h),int(new_w)

		# img = transform.resize(image , (new_w,new_h))
		img = transforms.functional.resize(image,(new_h,new_w))
		if flag==0:
			img = transforms.functional.pad(img,(pad_l,0,pad_r,0))
		else:
			img = transforms.functional.pad(img,(0,pad_l,0,pad_r))
		return img


def main():
	global args
	args = parser.parse_args()
	print(args)
	# fix random seeds
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	np.random.seed(args.seed)

	# CNN
	if args.verbose:
		print('Architecture: {}'.format(args.arch))
	
	model = models.__dict__[args.arch](out = args.num_classes, sobel=args.sobel)
	fd = int(model.top_layer.weight.size()[1])
	model.top_layer = None
	print("Parallelising ..")
	model.features = torch.nn.DataParallel(model.features)
	model.cuda()
	cudnn.benchmark = True

	print("Creating Optimizer ..")
	# create optimizer
	optimizer = torch.optim.SGD(
		filter(lambda x: x.requires_grad, model.parameters()),
		lr=args.lr,
		momentum=args.momentum,
		weight_decay=10**args.wd,
	)
	# define loss function
	criterion = nn.CrossEntropyLoss().cuda()

	# optionally resume from a checkpoint
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			args.start_epoch = checkpoint['epoch']
			# remove top_layer parameters from checkpoint
			for key in checkpoint['state_dict']:
				if 'top_layer' in key:
					del checkpoint['state_dict'][key]
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			print("=> loaded checkpoint '{}' (epoch {})"
				  .format(args.resume, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))

	# creating checkpoint repo
	exp_check = os.path.join(args.exp, 'Checkpoints')
	if not os.path.isdir(exp_check):
		os.makedirs(exp_check)

	# creating cluster assignments log
	cluster_log = Logger(os.path.join(args.exp, 'clusters'))

	# preprocessing of data
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])

	
	args.start_epoch = 0

	tra = [Resize(512),
		   # transforms.CenterCrop(224),
		   transforms.Resize(512),
		   transforms.RandomHorizontalFlip(p=0.5),
		   transforms.ToTensor(),
		   normalize
		   ]

	# load the data
	end = time.time()

	# clustering algorithm to use
	deepcluster = clustering.__dict__[args.clustering](args.nmb_cluster)
	dataset = datasets.ImageFolder(args.data , transform=transforms.Compose(tra))
	dataloader = torch.utils.data.DataLoader(dataset,batch_size=args.batch,
											 num_workers=args.workers,pin_memory=True)
	for epoch in range(args.start_epoch, args.epochs):
		
		end = time.time()


		# remove head
		model.top_layer = None
		##################################
		model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])#.half()
		##################################

		# ****************
		#get the features for the whole dataset
		features = compute_features(dataloader, model, len(dataset))
		# features = np.random.randn(14392,1920)

		# # cluster the features
		clustering_loss = deepcluster.cluster(features, verbose=args.verbose)

		# # assign pseudo-labels
		train_dataset = clustering.cluster_assign(deepcluster.images_lists,
												  dataset.imgs)

		# # uniformely sample per target
		sampler = UnifLabelSampler(int(args.reassign * len(train_dataset)),deepcluster.images_lists)
		# ****************
		
		# set last fully connected layer
		mlp = list(model.classifier.children())
		mlp.append(nn.ReLU(inplace=True).cuda())
		model.classifier = nn.Sequential(*mlp)
		# model.top_layer = nn.Linear(fd, len(deepcluster.images_lists)) 
		model.top_layer = nn.Linear(fd,6600)   #****************
		model.top_layer.weight.data.normal_(0, 0.01)
		model.top_layer.bias.data.zero_()
		model.top_layer.cuda()
		# train network with clusters as pseudo-labels
		end = time.time()
		loss = train(dataloader, model, criterion, optimizer, epoch)

		# print log
		if args.verbose:
			print('###### Epoch [{0}] ###### \n'
				  'Time: {1:.3f} s\n'
				  'Clustering loss: {2:.3f} \n'
				  'ConvNet loss: {3:.3f}'
				  .format(epoch, time.time() - end, clustering_loss, loss))
			try:
				nmi = normalized_mutual_info_score(
					clustering.arrange_clustering(deepcluster.images_lists),
					clustering.arrange_clustering(cluster_log.data[-1])
				)
				print('NMI against previous assignment: {0:.3f}'.format(nmi))
			except IndexError:
				pass
			print('####################### \n')
		# save running checkpoint
		torch.save({'epoch': epoch + 1,
					'arch': args.arch,
					'state_dict': model.state_dict(),
					'optimizer' : optimizer.state_dict()	},
				   os.path.join(args.exp, 'checkpoint.pth.tar'))
		if epoch%args.checkpoint == 0:
			torch.save({'epoch': epoch + 1,
						'arch': args.arch,
						'state_dict': model.state_dict(),
						'optimizer' : optimizer.state_dict()},
					   os.path.join(args.exp, 'Checkpoints','checkpoint_{0}_.pth.tar'.format(epoch+1)))

		# save cluster assignments
		cluster_log.log(deepcluster.images_lists)

def train(loader, model, crit, opt, epoch):
	"""Training of the CNN.
		Args:
			loader (torch.utils.data.DataLoader): Data loader
			model (nn.Module): CNN
			crit (torch.nn): loss
			opt (torch.optim.SGD): optimizer for every parameters with True
								   requires_grad in model except top layer
			epoch (int)
	"""
	batch_time = AverageMeter()
	losses = AverageMeter()
	data_time = AverageMeter()
	forward_time = AverageMeter()
	backward_time = AverageMeter()

	# switch to train mode
	model.train()

	# create an optimizer for the last fc layer
	optimizer_tl = torch.optim.SGD(
		model.top_layer.parameters(),
		lr=args.lr,
		weight_decay=10**args.wd,
	)

	end = time.time()
	pbar = tqdm(loader, ncols = 150)
	pbar.set_description("Epoch : {0}".format(epoch))
	for i, (input_var, target_var) in enumerate(pbar):
		# data_time.update(time.time() - end)

		# # save checkpoint
		# n = len(loader) * epoch + i
		# target = target.cuda(async=True)
		# print("Time taken for input sending to gpu")
		# start = time.time()
		input_var = input_var.cuda()
		target_var = target_var.cuda(async=True)
		# input_var = torch.autograd.Variable(input_tensor.cuda())
		# target_var = torch.autograd.Variable(target)
		# print(time.time() - start)
		# print("Time taken for forward")
		# start = time.time()
		output = model(input_var)
		# print(time.time() - start)
		# print("Time taken for loss calculation")
		# start = time.time()
		loss = crit(output, target_var)
		# print(time.time() - start)
		# record loss
		losses.update(loss.item(),input_var.size(0))

		# compute gradient and do SGD step
		# print("Time taken for backward")
		# start = time.time()
		opt.zero_grad()
		optimizer_tl.zero_grad()
		loss.backward()
		opt.step()
		optimizer_tl .step()
		# print(time.time() - start)

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()
		pbar.set_postfix(Loss = losses.val)

	return losses.avg

def compute_features(dataloader, model, N):
	if args.verbose:
		print("COMPUTING FEATURES ...")
	batch_time = AverageMeter()
	end = time.time()
	model.eval()
	# discard the label information in the dataloader
	
	pbar = tqdm(dataloader, ncols = 150)
	# pdb.set_trace()
	for i, (input_tensor, _) in enumerate(pbar):
		input_var = input_tensor.cuda()

		with torch.no_grad():
			aux = model(input_var).data.cpu().numpy()

		if i == 0:
			features = np.zeros((N, aux.shape[1])).astype('float32')

		# if i < len(dataloader) - 1:
		# 	features[i * args.batch: (i + 1) * args.batch] = aux.astype('float32')
		# else:
		# 	# special treatment for final batch
		# 	features[i * args.batch:] = aux.astype('float32')

		if i < len(dataloader) - 1:
			features[i * dataloader.batch_size : (i + 1) * dataloader.batch_size] = aux.astype('float32')
		else:
			# special treatment for final batch
			features[i * dataloader.batch_size:] = aux.astype('float32')

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()


		
	return features

if __name__ == '__main__':
	main()
