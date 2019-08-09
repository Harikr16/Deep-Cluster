#!/usr/bin/env python

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
import pdb
import numpy as np
import models
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import os 

def train(args, model, device, train_loader, optimizer, epoch,scheduler,writer):

	criterion = nn.CrossEntropyLoss().cuda()
	model.train()
	pbar = tqdm(train_loader)
	for batch_idx, (data, target) in enumerate(pbar):
		n_iter = epoch*len(train_loader) + batch_idx
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(data)
		loss = criterion(output,target)
		loss.backward()
		optimizer.step()
		writer.add_scalar('Loss/training',loss.item(),n_iter)

		# if batch_idx>2:
		# 	break
		

def test(args, model, device, test_loader,scheduler,writer,epoch):
	model.eval()
	test_loss = 0
	correct = 0
	pbar = tqdm(test_loader)
	with torch.no_grad():
		for batch_idx,(data, target) in enumerate(pbar):
			data, target = data.to(device), target.to(device)
			output = model(data)
			loss = F.nll_loss(output, target, reduction='sum')
			# loss = nn.CrossEntropyLoss(output,target,reduction='sum')
			test_loss += loss.item() # sum up batch loss
			pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
			correct += pred.eq(target.view_as(pred)).sum().item()

			# if batch_idx>2:
			# 	break

	test_loss /= len(test_loader.dataset)
	acc = correct/len(test_loader.dataset)
	scheduler.step(test_loss)
	tqdm.write("Validation accuracy : {}".format(acc))
	writer.add_scalar('Accuracy/Validation',acc,epoch)
	return acc

def make_weights_for_balanced_classes(images, nclasses):                        
	count = [0] * nclasses                                                      
	for item in images:                                                         
		count[item[1]] += 1                                                     
	weight_per_class = [0.] * nclasses                                      
	N = float(sum(count))                                                   
	for i in range(nclasses):                                                   
		weight_per_class[i] = N/float(count[i])                                 
	weight = [0] * len(images)                                              
	for idx, val in enumerate(images):                                          
		weight[idx] = weight_per_class[val[1]]                                  
	return weight 

def save_cpt(model,cpt_folder,n,optimizer = None,best = False):
	state_dict = {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':n}
	os.makedirs(os.path.join(cpt_folder,'supervised_learning'),exist_ok=True)
	if best:
		torch.save(state_dict,os.path.join(cpt_folder,'supervised_best.pt'))
	else:
		torch.save(state_dict,os.path.join(cpt_folder,'supervised_learning',str(n)+'.pt'))

class pretrained_model(nn.Module):
	def __init__(self,num_classes):
		super(pretrained_model,self).__init__()
		pre_model = torchvision.models.densenet121(pretrained = True)
		self.features = nn.Sequential(*list(pre_model.features.children())[:-1])
		self.pool = nn.AdaptiveMaxPool2d(1)
		self.classifier = nn.Linear(in_features=1024, out_features=num_classes, bias=True)

	def forward(self,x):
		x = self.features(x)
		x = self.pool(x)
		x = x.view(-1,1024)
		x = self.classifier(x)
		return x

def main():
	# Training settings
	parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
	parser.add_argument('--batch_size', type=int, default=64, metavar='N',
						help='input batch size for training (default: 64)')
	parser.add_argument('--epochs', type=int, default=10, metavar='N',
						help='number of epochs to train (default: 10)')
	parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
						help='learning rate (default: 0.01)')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
						help='random seed (default: 1)')
	parser.add_argument('--contd', type = int, default =1,help = "1 for continued training")	
	parser.add_argument('--num_classes',type= int,default=2,help = "number of classes")
	parser.add_argument('--resume',type=str,default="",help="enter checkpoint name to resume")
	parser.add_argument('--exp',type=str,default="Test/iprings",help="Folder name to save")
	parser.add_argument('--start_epoch',type=int,default=0,help="strat epoch")
	parser.add_argument('--size_of_img',type=int,default=224,help="size of resized image")
	parser.add_argument('--model_type',type=str,default="",help="type of pretrained model")
	args = parser.parse_args()

	torch.manual_seed(args.seed)
	print(args)
	device = torch.device("cuda")
	# train_root = "/media/photogauge/Data/Datasets/Iprings/train"
	# valid_root = "/media/photogauge/Data/Datasets/Iprings/test"
	train_root = r"D:\Datasets\Iprings\train"
	valid_root = r"D:\Datasets\Iprings\test"
	kwargs = {'num_workers': 0, 'pin_memory': True}
	tra = [transforms.Resize((args.size_of_img,args.size_of_img)),
		   transforms.RandomHorizontalFlip(p=0.5),
		   transforms.ToTensor(),
		   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		   ]
	tra_1 = [transforms.Resize((args.size_of_img,args.size_of_img)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		   	]       
	train_dataset = datasets.ImageFolder(train_root , transform=transforms.Compose(tra))
	valid_dataset = datasets.ImageFolder(valid_root, transform=transforms.Compose(tra_1))

	weights = make_weights_for_balanced_classes(train_dataset.imgs,args.num_classes)
	weights = torch.DoubleTensor(weights)                                       
	sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))                     

	train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size, sampler = sampler)
	valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=args.batch_size*10, shuffle=True)
	


	if args.model_type == "deepcluster":
		model = models.__dict__["densenet"](out = args.num_classes, sobel=False)

		print("Parallelising ..")
		model.features = torch.nn.DataParallel(model.features)
		model.cuda()
		cudnn.benchmark = True
		
		if args.resume:
			if os.path.isfile(args.resume):
				print("=> loading checkpoint '{}'".format(args.resume))
				checkpoint = torch.load(args.resume)
				# args.start_epoch = 47
				# model = torch.load(args.resume)
				##########################
				if args.contd==0:
					#remove top_layer parameters from checkpoint
					model.top_layer = None
					import copy
					new_cpt = copy.deepcopy(checkpoint)
					for key in checkpoint['state_dict']:
						if 'top_layer' in key:
							del new_cpt['state_dict'][key]
					model.load_state_dict(new_cpt['state_dict'])
					model.top_layer = torch.nn.Linear(1920, args.num_classes).cuda()
				###########################
				##########################

				else:
					
					args.start_epoch = checkpoint['epoch']+1
					model.load_state_dict(checkpoint['model'])
					optimizer.load_state_dict(checkpoint['optimizer'])
					# pdb.set_trace()   
					# save_cpt(model,'Runs',47)
					# checkpoint = torch.load(args.resume)
					# model = checkpoint['model']
				###########################         
					# args.start_epoch = checkpoint['epoch']+1
					# model = checkpoint['model']
					#################################
			else:
				print("=> no checkpoint found at '{}'".format(args.resume))

	elif args.model_type == "pretrained":
		model = pretrained_model(num_classes=2).cuda()

		print("Parallelising ..")
		model.features = torch.nn.DataParallel(model.features)
		model.cuda()
		cudnn.benchmark = True
	
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
	writer = SummaryWriter(os.path.join(args.exp,'Supervised_Learning_Logs'))
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True,eps=1e-08)

	best_acc = 0
	for epoch in range(args.start_epoch, args.epochs + 1):

		train(args, model, device, train_loader, optimizer, epoch,scheduler,writer)
		acc = test(args, model, device, valid_loader,scheduler,writer,epoch)

		save_cpt(model,args.exp,epoch,optimizer,best=False)
		if best_acc<acc:
			best_acc = acc
			tqdm.write("Best Validation Accuracy : {}".format(best_acc))
			save_cpt(model,args.exp,epoch,optimizer,best=True)

		
if __name__ == '__main__':
	main()
