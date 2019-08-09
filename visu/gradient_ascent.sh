# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

MODEL='~/Codes/DC/test/exp/checkpoints/checkpoint_0.pth.tar'
ARCH='resnet'
EXP='~/Codes/DC/test/exp'
CONV=6

python gradient_ascent.py --model ${MODEL} --exp ${EXP} --conv ${CONV} --arch ${ARCH}
