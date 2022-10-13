## Inference with pre-trained models

We provide testing scripts to evaluate pre-trained models on a whole dataset,
as well as some task-specific image demos.

### Test a pre-trained model

MMEditing implements **distributed** testing with `MMDistributedDataParallel`.

#### Test with single/multiple GPUs

You can use the following commands to test a pre-trained model with single/multiple GPUs.

```shell
# single-gpu testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--save-path ${IMAGE_SAVE_PATH}]

# multi-gpu testing
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--save-path ${IMAGE_SAVE_PATH}]
```

For example,

```shell
# single-gpu testing
python tools/test.py configs/example_config.py work_dirs/example_exp/example_model_20200202.pth --out work_dirs/example_exp/results.pkl

# multi-gpu testing
./tools/dist_test.sh configs/example_config.py work_dirs/example_exp/example_model_20200202.pth --save-path work_dirs/example_exp/results/
```

#### Test with Slurm

If you run MMEditing on a cluster managed with [slurm](https://slurm.schedmd.com/), you can use the script `slurm_test.sh`. (This script also supports single machine testing.)

```shell
[GPUS=${GPUS}] ./tools/slurm_test.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${CHECKPOINT_FILE}
```

Here is an example of using 8 GPUs to test an example model on the 'dev' partition with job name 'test'.

```shell
GPUS=8 ./tools/slurm_test.sh dev test configs/example_config.py work_dirs/example_exp/example_model_20200202.pth
```

You can check [slurm_test.sh](https://github.com/open-mmlab/mmediting/blob/master/tools/slurm_test.sh) for full arguments and environment variables.

#### Optional arguments

- `--out`: Specify the filename of the output results in pickle format. If not given, the results will not be saved to a file.
- `--save-path`: Specify the path to store edited images. If not given, the images will not be saved.
- `--seed`: Random seed during testing. This argument is used for fixed results in some tasks such as inpainting.
- `--deterministic`: Related to `--seed`, this argument decides whether to set deterministic options for CUDNN backend. If specified, it will set `torch.backends.cudnn.deterministic` to True and `torch.backends.cudnn.benchmark` to False.
- `--cfg-options`: If specified, the key-value pair optional cfg will be merged into config file.

Note: Currently, we do NOT use `--eval` argument like [MMDetection](https://github.com/open-mmlab/mmdetection) to specify evaluation metrics. The evaluation metrics are given in the config files (see [config.md](config.md)).

## Train a model

MMEditing implements **distributed** training with `MMDistributedDataParallel`.

All outputs (log files and checkpoints) will be saved to the working directory,
which is specified by `work_dir` in the config file.

By default we evaluate the model on the validation set after several iterations, you can change the evaluation interval by adding the interval argument in the training config.

```python
evaluation = dict(interval=1e4, by_epoch=False)  # This evaluates the model per 1e4 iterations.
```

### Train with single/multiple GPUs

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

Optional arguments are:

- `--no-validate` (**not suggested**): By default, the codebase will perform evaluation every k iterations during the training. To disable this behavior, use `--no-validate`.
- `--work-dir ${WORK_DIR}`: Override the working directory specified in the config file.
- `--resume-from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.
- `--cfg-options`: If specified, the key-value pair optional cfg will be merged into config file.

Difference between `resume-from` and `load-from`:
`resume-from` loads both the model weights and optimizer status, and the iteration is also inherited from the specified checkpoint. It is usually used for resuming the training process that is interrupted accidentally.
`load-from` only loads the model weights and the training iteration starts from 0. It is usually used for fine-tuning.

#### Train with multiple nodes

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

### Launch multiple jobs on a single machine

If you launch multiple jobs on a single machine, e.g., 2 jobs of 4-GPU training on a machine with 8 GPUs,
you need to specify different ports (29500 by default) for each job to avoid communication conflict.

If you use `dist_train.sh` to launch training jobs, you can set the port in commands.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh ${CONFIG_FILE} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh ${CONFIG_FILE} 4
```

If you launch training jobs with Slurm, you need to modify the config files (usually the 6th line from the bottom in config files) to set different communication ports.

In `config1.py`,

```python
dist_params = dict(backend='nccl', port=29500)
```

In `config2.py`,

```python
dist_params = dict(backend='nccl', port=29501)
```

Then you can launch two jobs with `config1.py` ang `config2.py`.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py ${WORK_DIR}
CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py ${WORK_DIR}
```
