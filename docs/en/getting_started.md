# Getting Started

This page provides basic tutorials about the usage of MMEditing.
For installation instructions, please see [install.md](install.md).

## Prepare datasets

It is recommended to symlink the dataset root to `$MMEditing/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

[Inpainting](https://mmediting.readthedocs.io/en/latest/_tmp/inpainting_datasets.html)

[Matting](https://mmediting.readthedocs.io/en/latest/_tmp/matting_datasets.html)

[Restoration](https://mmediting.readthedocs.io/en/latest/_tmp/sr_datasets.html)

[Generation](https://mmediting.readthedocs.io/en/latest/_tmp/generation_datasets.html)

## Inference with pretrained models

We provide testing scripts to evaluate a whole dataset,
as well as some task-specific image demos.

### Test a dataset

MMEditing implements **distributed** testing with `MMDistributedDataParallel`.

#### Test with single/multiple GPUs

You can use the following commands to test a dataset with single/multiple GPUs.

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
./tools/dist_test.sh configs/example_config.py work_dirs/example_exp/example_model_20200202.pth 2 --save-path work_dirs/example_exp/results/
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

### Image demos

We provide some task-specific demo scripts to test a single image.

#### Inpainting

You can use the following commands to test a pair of image and trimap.

```shell
python demo/inpainting_demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${MASKED_IMAGE_FILE} ${MASK_FILE} ${SAVE_FILE} [--imshow] [--device ${GPU_ID}]
```

If `--imshow` is specified, the demo will also show image with opencv. Examples:

```shell
python demo/inpainting_demo.py configs/inpainting/global_local/gl_256x256_8x12_celeba.py xxx.pth tests/data/image/celeba_test.png tests/data/image/bbox_mask.png tests/data/pred/inpainting_celeba.png
```

The predicted inpainting result will be save in `tests/data/pred/inpainting_celeba.png`.

#### Matting

You can use the following commands to test a pair of image and trimap.

```shell
python demo/matting_demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${IMAGE_FILE} ${TRIMAP_FILE} ${SAVE_FILE} [--imshow] [--device ${GPU_ID}]
```

If `--imshow` is specified, the demo will also show image with opencv. Examples:

```shell
python demo/matting_demo.py configs/mattors/dim/dim_stage3_v16_pln_1x1_1000k_comp1k.py work_dirs/dim_stage3/latest.pth tests/data/merged/GT05.jpg tests/data/trimap/GT05.png tests/data/pred/GT05.png
```

The predicted alpha matte will be save in `tests/data/pred/GT05.png`.

#### Restoration

You can use the following commands to test an image for restoration.

```shell
python demo/restoration_demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${IMAGE_FILE} ${SAVE_FILE} [--imshow] [--device ${GPU_ID}]
```

If `--imshow` is specified, the demo will also show image with opencv. Examples:

```shell
python demo/restoration_demo.py configs/restorers/esrgan/esrgan_x4c64b23g32_g1_400k_div2k.py work_dirs/esrgan_x4c64b23g32_1x16_400k_div2k/latest.pth tests/data/lq/baboon_x4.png demo/demo_out_baboon.png
```

The restored image will be save in `demo/demo_out_baboon.png`.

#### Generation

```shell
python demo/generation_demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${IMAGE_FILE} ${SAVE_FILE} [--unpaired-path ${UNPAIRED_IMAGE_FILE}] [--imshow] [--device ${GPU_ID}]
```

If `--unpaired-path` is specified (used for CycleGAN), the model will perform unpaired image-to-image translation. If `--imshow` is specified, the demo will also show image with opencv. Examples:

Paired:

```shell
python demo/generation_demo.py configs/example_config.py work_dirs/example_exp/example_model_20200202.pth demo/demo.jpg demo/demo_out.jpg
```

Unpaired (also show image with opencv):

```shell
python demo/generation_demo.py configs/example_config.py work_dirs/example_exp/example_model_20200202.pth demo/demo.jpg demo/demo_out.jpg --unpaired-path demo/demo_unpaired.jpg --imshow
```

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
`load-from` only loads the model weights and the training iteration starts from 0. It is usually used for finetuning.

### Train with Slurm

If you run MMEditing on a cluster managed with [slurm](https://slurm.schedmd.com/), you can use the script `slurm_train.sh`. (This script also supports single machine training.)

```shell
[GPUS=${GPUS}] ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR}
```

Here is an example of using 8 GPUs to train an inpainting model on the dev partition.

```shell
GPUS=8 ./tools/slurm_train.sh dev places_256 configs/inpainting/gl_places.py /nfs/xxxx/gl_places_256
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

## Useful tools

We provide lots of useful tools under `tools/` directory.

### Get the FLOPs and params (experimental)

We provide a script adapted from [flops-counter.pytorch](https://github.com/sovrasov/flops-counter.pytorch) to compute the FLOPs and params of a given model.

```shell
python tools/get_flops.py ${CONFIG_FILE} [--shape ${INPUT_SHAPE}]
```

For example,

```shell
python tools/get_flops.py configs/resotorer/srresnet.py --shape 40 40
```

You will get the result like this.

```
==============================
Input shape: (3, 40, 40)
Flops: 4.07 GMac
Params: 1.52 M
==============================
```

**Note**: This tool is still experimental and we do not guarantee that the number is correct. You may well use the result for simple comparisons, but double check it before you adopt it in technical reports or papers.

(1) FLOPs are related to the input shape while parameters are not. The default input shape is (1, 3, 250, 250).
(2) Some operators are not counted into FLOPs like GN and custom operators.
You can add support for new operators by modifying [`mmcv/cnn/utils/flops_counter.py`](https://github.com/open-mmlab/mmcv/blob/master/mmcv/cnn/utils/flops_counter.py).

### Publish a model

Before you upload a model to AWS, you may want to
(1) convert model weights to CPU tensors, (2) delete the optimizer states and
(3) compute the hash of the checkpoint file and append the hash id to the filename.

```shell
python tools/publish_model.py ${INPUT_FILENAME} ${OUTPUT_FILENAME}
```

E.g.,

```shell
python tools/publish_model.py work_dirs/example_exp/latest.pth example_model_20200202.pth
```

The final output filename will be `example_model_20200202-{hash id}.pth`.
