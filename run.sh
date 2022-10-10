#!/usr/bin/env bash

GPUS=4 GPUS_PER_NODE=4 ./tools/slurm_train.sh \
mm_lol \
realgan_test \
configs_ceph/real_esrgan/realesrgan_c64b23g32_4xb12-lr1e-4-400k_df2k-ost.py \
# checkpoint/RealESRGAN_x4plus_revised.pth
# train_log \
# --resume