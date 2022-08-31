# Testing and Training in MMEditing

In this section, we introduce how to test and train models in MMEditing.

## Prerequisite

Users need to [prepare dataset](../user_guides/datasets/dataset_prepare.md) first to train and test models in MMEditing.

## Test a pre-trained model in MMEditing

MMEditing supports multiple ways to test a pre-trained model in MMEditing:

1. [Test with single GPUs](#test-with-single-gpus)
2. [Test with multiple GPUs](#test-with-multiple-gpus)
3. [Test with Slurm](#test-with-slurm)

### Test with single GPUs

You can use the following commands to test a pre-trained model with single GPUs.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}
```

For example,

```shell
python tools/test.py configs/example_config.py work_dirs/example_exp/example_model_20200202.pth
```

### Test with multiple GPUs

MMEditing supports testing with multiple GPUs,
which can largely save your time in testing models.
You can use the following commands to test a pre-trained model with multiple GPUs.

```shell
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM}
```

For example,

```shell
./tools/dist_test.sh configs/example_config.py work_dirs/example_exp/example_model_20200202.pth
```

### Test with Slurm

If you run MMEditing on a cluster managed with [slurm](https://slurm.schedmd.com/), you can use the script `slurm_test.sh`. (This script also supports single machine testing.)

```shell
[GPUS=${GPUS}] ./tools/slurm_test.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${CHECKPOINT_FILE}
```

Here is an example of using 8 GPUs to test an example model on the 'dev' partition with job name 'test'.

```shell
GPUS=8 ./tools/slurm_test.sh dev test configs/example_config.py work_dirs/example_exp/example_model_20200202.pth
```

You can check [slurm_test.sh](../../../tools/slurm_test.sh) for full arguments and environment variables.

## Train a model in MMEditing

MMEditing supports multiple ways of training:

1. [Train with a single GPU](#train-with-a-single-gpu)
2. [Train with a single node multiple GPUs](#train-with-a-single-node-multiple-gpus)
3. [Train with multiple nodes](#train-with-multiple-nodes)
4. [Train with Slurm](#train-with-slurm)

Specifically, all outputs (log files and checkpoints) will be saved to the working directory,
which is specified by `work_dir` in the config file.

### Train with a single GPU

```shell
CUDA_VISIBLE=0 python tools/train.py configs/example_config.py --work-dir work_dirs/example
```

### Train with a single node multiple GPUs

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

### Train with multiple nodes

To launch distributed training on multiple machines, which can be accessed via IPs, run following commands:

On the first machine:

```shell
NNODES=2 NODE_RANK=0 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR tools/dist_train.sh $CONFIG $GPUS
```

On the second machine:

```shell
NNODES=2 NODE_RANK=1 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR tools/dist_train.sh $CONFIG $GPUS
```

To speed up network communication, high speed network hardware, such as Infiniband, is recommended.
Please refer to [PyTorch docs](https://pytorch.org/docs/1.11/distributed.html#launch-utility) for more information.

### Train with Slurm

If you run MMEditing on a cluster managed with [slurm](https://slurm.schedmd.com/), you can use the script `slurm_train.sh`. (This script also supports single machine training.)

```shell
[GPUS=${GPUS}] ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR}
```

Here is an example of using 8 GPUs to train an inpainting model on the dev partition.

```shell
GPUS=8 ./tools/slurm_train.sh dev configs/inpainting/gl_places.py /nfs/xxxx/gl_places_256
```

You can check [slurm_train.sh](https://github.com/open-mmlab/mmediting/blob/master/tools/slurm_train.sh) for full arguments and environment variables.

### Optional arguments

- `--amp`: This argument is used for fixed-precision training.
- `--resume`: This argument is used for auto resume if the training is aborted.
