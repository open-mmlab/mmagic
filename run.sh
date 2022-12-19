GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=16 ./tools/slurm_train.sh \
aide_lol \
ifrnet \
configs/ifrnet/ifrnet_in2out7_8xb4_gopro.py \
work_dirs/ifrnet_in2out7_8xb4_gopro \
--resume
