#!/usr/bin/env bash

GPUS=8 GPUS_PER_NODE=8 ./tools/slurm_train.sh \
mm_lol \
nafnet_train \
configs/nafnet/nafnet_c64eb11128mb1db1111_lr1e-3_400k_gopro.py \
work_dirs/nafnet_c64eb11128mb1db1111_lr1e-3_400k_gopro \
# --resume
