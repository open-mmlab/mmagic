#!/usr/bin/env bash

# GPUS=8 GPUS_PER_NODE=8 ./tools/slurm_train.sh \
# mm_lol \
# nafnet_train \
# configs/nafnet/nafnet_c64eb11128mb1db1111_lr1e-3_400k_gopro.py \
# work_dirs/nafnet_c64eb11128mb1db1111_lr1e-3_400k_gopro \
# --resume


GPUS=4 GPUS_PER_NODE=4 ./tools/slurm_test.sh \
mm_lol \
nafnet_test \
configs/nafnet/nafnet_c64eb2248mb12db2222_lr1e-3_400k_sidd.py \
checkpoint/NAFNet-SIDD-midc64.pth