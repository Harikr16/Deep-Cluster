# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

# DIR="/home/photogauge/Datasets/Gears/"
# DIR="/media/photogauge/Data/Datasets/Iprings/"
DIR="/media/photogauge/7E30E7C830E7858D/Dataset/Naturalist/inaturalist-2019-fgvc6"
ARCH="densenet"
# ARCH="alexnet"
LR=5e-4
WD=-5
WORKERS=4
# EXP="Runs/Naturalist_DenseNet_prog_fp16_no_mixup_no_scheduler_loaded_opt_sampled"
# EXP="Runs/ResNet152_pretrained_kmeans_k_20_sobel"
EXP="Runs/DenseNet_fp16_supconv_mixup_nosobel_contd_transforms_added"
# EXP="Test/exp"
EPOCHS=200


START_EPOCH=0
BATCH=16
MOMENTUM=0.9
# RESUME="/media/photogauge/Data/Codes/Ubuntu/Codes/DC/Runs/ResNet152_pretrained_kmeans_k_20_sobel/checkpoint.pth.tar"
RESUME="/media/photogauge/Data/Codes/Ubuntu/Codes/DC/Runs/DenseNet_fp16_supconv_mixup_nosobel_contd_transforms_added/Supervised_checkpoints/42.pt" 
# RESUME="/media/photogauge/Data/Codes/Ubuntu/Codes/DC/test_model.pt"
# RESUME="/media/photogauge/Data/Codes/Ubuntu/Codes/DC/Runs/47.pt"
CHECKPOINT=1
SEED=31
NUM_CLASSES=1010
SPLIT=0
CONTD=1
FP16=0
MIXUP="True"
# MODE="TRAIN"
MODE="TEST"
# SUP_CONV="False"
SUP_CONV="True"
python -W ignore supervised_learner.py --data ${DIR} --exp ${EXP} --arch ${ARCH} --lr ${LR} --wd ${WD}  --workers ${WORKERS}  \
    --num_classes ${NUM_CLASSES} --batch ${BATCH} --momentum ${MOMENTUM} --sup_conv ${SUP_CONV} --resume ${RESUME}\
    --seed ${SEED} --split ${SPLIT} --contd ${CONTD} --fp16 ${FP16} --mode ${MODE} --mixup ${MIXUP} --verbose #--sobel 

    #--resume ${RESUME}
