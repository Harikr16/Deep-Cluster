# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

# DIR="/home/photogauge/Datasets/Gears/"
DIR="/media/photogauge/Data/Mazhar/Demo_2"
ARCH="densenet"
# ARCH="alexnet"
LR=1e-5
WD=-5
K=1000
# K=200
WORKERS=0
EXP="Runs/IPRINGS_densenet_512_no_sobel_AFTER_200"
# EXP="Test/exp"
EPOCHS=600

CLUSTERING='Kmeans'
REASSIGN=1.
START_EPOCH=0
BATCH=12
MOMENTUM=0.9
# RESUME="/home/photogauge/Codes/DC/Runs/ResNet152_pretrained_kmeans_k_20_sobel/Checkpoints/checkpoint_26_.pth.tar"
RESUME="/media/photogauge/Data/Codes/Ubuntu/Codes/DC/Runs/IPRINGS_densenet_512_no_sobel/checkpoint.pth.tar"
# RESUME=""
CHECKPOINT=1
SEED=31
NUM_CLASSES=1010
UPDATE_LOADER=4
#

mkdir -p ${EXP}

# psudo CUDA_VISIBLE_DEVICES=1 python main.py "/home/photogauge/Datasets/Gears/" --exp "~/Codes/DC/test/exp" --arch "resnet152" --lr 0.05 --wd -5 --k 10000 --sobel --verbose --workers 1
python -W ignore main_iprings.py ${DIR} --exp ${EXP} --arch ${ARCH} --lr ${LR} --wd ${WD} --k ${K}  --workers ${WORKERS} \
	--num_classes ${NUM_CLASSES} --clustering ${CLUSTERING} --reassign ${REASSIGN} --batch ${BATCH} --momentum ${MOMENTUM} --resume ${RESUME}\
	--checkpoint ${CHECKPOINT} --seed ${SEED} --update_loader ${UPDATE_LOADER} --verbose 

	#--resume ${RESUME}