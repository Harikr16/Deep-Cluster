import torch 
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data.dataset import Dataset

import numpy as np
import pdb
import json
from tqdm import tqdm
import os
import argparse
import models


from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator, DALIClassificationIterator
import nvidia.dali.ops as ops
import nvidia.dali.types as types

def ram_use():
	import psutil
	pid = os.getpid()
	py = psutil.Process(pid)
	memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
	return memoryUse

def get_subset_sampler(dataset,split_ratio):
	""" Get Train and valid samplers from a single dataset """
	# SPLIT RATIO is the size of validation dataset
	
	dataset_size = len(dataset)
	indices = list(range(dataset_size))
	split = int(np.floor(split_ratio * dataset_size))
	np.random.shuffle(indices)
	train_indices, val_indices = indices[split:], indices[:split]
	samplers = (SubsetRandomSampler(train_indices),SubsetRandomSampler(val_indices))
	return samplers


def validate(model,valid_loader):

	with torch.no_grad():
		loader = tqdm(valid_loader, ncols=75)
		correct = np.array([])
		for data,target in loader:

			data,target = data.cuda(),target.cuda()
			# pdb.set_trace()
			output = model(data)

		correct = np.concatenate((correct,torch.eq(torch.argmax(output,dim=-1),target).cpu().numpy()))

	accuracy = correct.mean()
	tqdm.write("Validation Accuracy : {0:2.3f}".format(accuracy*100))


def train(model,optimizer,criterion,dataloaders,args):

	nepochs = args.epochs
	# train_loader = dataloaders[0]
	# valid_loader =dataloaders[1]

	
	for n in range(nepochs):
		loader = tqdm(dataloaders, ncols=75)
		loader.set_description("Epoch : {0}".format(n+1))
		correct = np.array([])
		# mean = np.array([0,0,0])
		# std = np.array([0,0,0])
		# samples = 0
		for data in loader:
			if n>0:
				pdb.set_trace()

			# data,target = data['data'].cuda(),data['label'].cuda()

			# optimizer.zero_grad()
			# output = model(data)
			# # pdb.set_trace()
			# loss = criterion(output,target)

			# loss.backward()
			# optimizer.step()

			# # correct = np.concatenate((correct,torch.eq(torch.argmax(output,dim=-1),target).cpu().numpy()))
			# correct = torch.eq(torch.argmax(output,dim=-1),target).cpu().numpy()
			# # print(correct)
			# loader.set_postfix(Train_loss = loss.item(), Train_accuracy = float(correct.sum())/len(data)*100)
			pass
			# pdb.set_trace()
		# 	for i in range(3):
		# 		mean[i]+= data.mean(-1).mean(-1).mean(0)[i]
		# 		std[i]+= data.std(-1).std(-1).std(0)[i]

		# 	samples+=data.size()[0]
		# print("Mean : {0}".format(mean/samples))
		# print("STD : {0}".format(mean/samples))
		# validate(model,valid_loader)


class Naturalist_Dset(datasets.ImageFolder):

	def __init__(self,root,annotation_fname,transform = None):
		super(Naturalist_Dset,self).__init__(root=root,transform= transform)

		self.transforms = transform
		with open(annotation_fname,'r') as f:
			data = json.load(f)

		self.image_name_id = {}
		for elements in data['images']:			
			self.image_name_id[os.path.join(os.path.dirname(root),elements['file_name'])] = elements['id']

		self.image_id_anno = {}		
		for elements in data['annotations']:
			self.image_id_anno[elements['id']] = elements['category_id']

	def __getitem__(self,index):

		path,target = self.imgs[index]

		try :
			target = self.image_id_anno[self.image_name_id[path]]
		except:
			index = index - 1 if index > 0 else index + 1 
			return self.__getitem__(index)

		img = self.loader(path)

		if self.transforms is not None:
			img = self.transforms(img)

		return (img,target)


	def __len__(self): 

		return len(self.imgs)


class NetPipeline(Pipeline):
	def __init__(self, image_dir, batch_size, num_threads, device_id, exec_async=True):
		super(NetPipeline, self).__init__(batch_size, num_threads, device_id, seed = 12, exec_async=exec_async)
		self.input = ops.FileReader(file_root = image_dir, random_shuffle = True, initial_fill = 21)
		self.decode = ops.nvJPEGDecoder(device = "mixed", output_type = types.RGB)
		self.resize = ops.Resize(device = "gpu", resize_x=512, resize_y = 512)
		# self.centerCrop = ops.Crop(device = "gpu", crop=(224,224))
		self.norm = ops.NormalizePermute(device = "gpu",
											height = 512,
											width = 512,
											mean = [x*255 for x in [0.485, 0.456, 0.406]],
											std = [x*255 for x in [0.229, 0.224, 0.225]])
	
	def define_graph(self):
		jpegs, labels = self.input()
		images = self.decode(jpegs)
		images = self.resize(images)
		# images = self.centerCrop(images)
		images = self.norm(images)
		
		# images are on the GPU
		return (images, labels)
			
def main(args):

	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	np.random.seed(args.seed)

	cpt_log = args.checkpoint
	# output = "Resu"
	cpt_folder = args.exp +"/Supervised_checkpoints/"
	if not os.path.exists(cpt_folder):
		os.makedirs(cpt_folder)

	# Data Loading	

	N = 2 # number of GPUs
	BATCH_SIZE = args.batch  # 128, batch size per GPU
	ITERATIONS = args.epochs
	NUM_THREADS = args.workers

	train_dir = '/media/photogauge/Data/Datasets/Naturalist/inaturalist-2019-fgvc6/train_val2019/Birds'

	pipes = [NetPipeline(image_dir=train_dir, batch_size=BATCH_SIZE, num_threads=NUM_THREADS, device_id=device_id,exec_async = False) for device_id in range(N)]
	pipes[0].build()
	train_iter = DALIGenericIterator(pipes, ['data', 'label'], pipes[0].epoch_size().popitem()[1])




	# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
	# 								 std=[0.229, 0.224, 0.225])
	# tra = [
	# 		transforms.Resize((512,512)),
	# 		# transforms.CenterCrop(224),
	# 	   transforms.ToTensor(),
	# 	   normalize
	# 	   ]

	# train_json = "/media/photogauge/Data/Datasets/Naturalist/inaturalist-2019-fgvc6/train2019.json"
	# valid_json = "/media/photogauge/Data/Datasets/Naturalist/inaturalist-2019-fgvc6/val2019.json"
	# print("Creating Dataset")
	# train_dataset = Naturalist_Dset(args.data + '/train_val2019',annotation_fname = train_json,transform=transforms.Compose(tra))
	# valid_dataset = Naturalist_Dset(args.data + '/train_val2019',annotation_fname = valid_json,transform=transforms.Compose(tra))
	
	# #sampler = get_subset_sampler(dataset = dataset,split_ratio = args.split)
	# print("Creating Loader")
	# train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle = True,#sampler = sampler[0],
	# 										 batch_size=args.batch,
	# 										 num_workers=args.workers,
	# 										 pin_memory=True)
	# valid_dataloader = torch.utils.data.DataLoader(valid_dataset, shuffle=True,#sampler = sampler[1],
	# 										 batch_size=args.batch*5,
	# 										 num_workers=args.workers,
	# 										 pin_memory=True)


	model = models.__dict__[args.arch](out = args.num_classes, sobel=args.sobel)
	print("Parallelising ..")
	model.features = torch.nn.DataParallel(model.features)
	model.cuda()
	cudnn.benchmark = True

	print("Creating Optimizer ..")
	# create optimizer
	# optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=10**args.wd, amsgrad=False)
	optimizer = torch.optim.SGD(
		filter(lambda x: x.requires_grad, model.parameters()),
		lr=args.lr,
		momentum=args.momentum,
		weight_decay=10**args.wd,
	)
	criterion = nn.CrossEntropyLoss().cuda()

	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			# args.start_epoch = checkpoint['epoch']
			# remove top_layer parameters from checkpoint
			model.top_layer = None
			for key in checkpoint['state_dict']:
				if 'top_layer' in key:
					del checkpoint['state_dict'][key]
			model.load_state_dict(checkpoint['state_dict'])
			# optimizer.load_state_dict(checkpoint['optimizer'])
			print("=> loaded checkpoint '{}' (epoch {})"
				  .format(args.resume, checkpoint['epoch']))
			model.top_layer = torch.nn.Linear(2048, args.num_classes).cuda()
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))

	# dataloaders = [train_dataloader,valid_dataloader]
	train(model,optimizer,criterion,train_iter,args)





# python supervised_learner.py --dir="~/Datasets/Gears/" --resume="~/Codes/DC/Runs/ResNet152_pretrained_kmeans_k_20/Checkpoints/checkpoint_196_.pth.tar"  --epochs= 50 --exp= --sobel


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster SUpervised Learning')
	parser.add_argument('data', metavar='DIR', help='path to dataset')
	parser.add_argument('--arch', '-a', type=str, metavar='ARCH',
						choices=['alexnet', 'vgg16','resnet'], default='resnet',
						help='CNN architecture (default: alexnet)')
	parser.add_argument('--sobel', action='store_true', help='Sobel filtering')
	parser.add_argument('--lr', default=0.05, type=float,
						help='learning rate (default: 0.05)')
	parser.add_argument('--wd', default=-5, type=float,
						help='weight decay pow (default: -5)')
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
	parser.add_argument('--split', type=float, default=.25, help='Ratio of valid dataset to train dataset')

	args = parser.parse_args()
	main(args)
