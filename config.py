DIR=r"C:/Dataset/Naturalist/inaturalist-2019-fgvc6"
ARCH="densenet"
LR=1e-1
WD=-5
WORKERS=12
"Runs/Naturalist_DenseNet_prog_fp16_no_mixup_no_scheduler_loaded_opt_sampled"
EXP= r"D:\Codes\Ubuntu\Codes\DC\Runs/Naturalist_DenseNet_prog_fp16_no_mixup_no_scheduler_loaded_opt_sampled"
EPOCHS=200
START_EPOCH=0
BATCH=16
MOMENTUM=0.9
# RESUME="/home/photogauge/Codes/DC/Runs/Naturalist/Checkpoints/checkpoint_31_.pth.tar" 
#/media/photogauge/Data/Codes/Ubuntu/Codes/DC/Runs/Naturalist_DenseNet_prog_fp16_no_mixup_no_scheduler_loaded_opt/Supervised_checkpoints/16.pt
RESUME= r"D:\Codes\Ubuntu\Codes\DC\Runs\Naturalist_DenseNet_prog_fp16_no_mixup_no_scheduler_loaded_opt_sampled\Supervised_checkpoints/35.pt"
CHECKPOINT=1
SEED=31
NUM_CLASSES=1010
SPLIT=0
CONTD=1
FP16=0
MODE="TRAIN"