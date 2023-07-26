# 教程 4：在MMagic环境下训练与测试

在该部分中，您将学到如何在MMagic环境下完成训练与测试

我们提供如下教程：

- [预先准备](#预先准备)
- [在MMagic中测试模型](#在MMagic中测试模型)
  - [在单个GPU上测试](#在单个GPU上测试)
  - [在多个GPU上测试](#在多个GPU上测试)
  - [在Slurm上测试](#在Slurm上测试)
  - [使用特定指标进行测试](#使用特定指标进行测试)
- [在MMagic中训练模型](#在MMagic中训练模型)
  - [在单个GPU上训练](#在单个GPU上训练)
  - [在多个GPU上训练](#在多个GPU上训练)
  - [在多个节点上训练](#在多个节点上训练)
  - [在Slurm上训练](#在Slurm上训练)
  - [使用特定的评估指标进行训练](#使用特定的评估指标进行训练)

## 预先准备

用户需要首先 [准备数据集](../user_guides/dataset_prepare.md) 从而能够在MMagic环境中训练和测试。

## 在MMagic中测试模型

### 在单个GPU上测试

您可以通过如下命令使用单个GPU来测试预训练模型。

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}
```

例如：

```shell
python tools/test.py configs/example_config.py work_dirs/example_exp/example_model_20200202.pth
```

### 在多个GPU上测试

MMagic支持使用多个GPU测试，能够极大地节约模型测试时间。
可以通过如下命令使用多个GPU来测试预训练模型。

```shell
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM}
```

例如：

```shell
./tools/dist_test.sh configs/example_config.py work_dirs/example_exp/example_model_20200202.pth
```

### 在Slurm上测试

如果您在由 [slurm](https://slurm.schedmd.com/) 管理的集群上运行MMagic，可以使用脚本`slurm_test.sh`。（此脚本还支持单机测试。）

```shell
[GPUS=${GPUS}] ./tools/slurm_test.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${CHECKPOINT_FILE}
```

下面是一个使用8个GPU在“dev”分区上测试一个示例模型的例子，作业名称为“test”。

```shell
GPUS=8 ./tools/slurm_test.sh dev test configs/example_config.py work_dirs/example_exp/example_model_20200202.pth
```

您可以检查 [slurm_test.sh](../../../tools/slurm_test.sh) 以获取完整的参数和环境变量。

### 使用特定指标进行测试

MMagic 提供各种评**估值指标**，例如：MS-SSIM、SWD、IS、FID、Precision&Recall、PPL、Equivarience、TransFID、TransIS等。
我们在[tools/test.py](https://github.com/open-mmlab/mmagic/tree/main/tools/test.py)中为所有模型提供了统一的评估脚本。
如果用户想用一些指标来评估他们的模型，你可以像这样将 `metrics` 添加到你的配置文件中:

```python
# 在文件 configs/styleganv2/stylegan2_c2_ffhq_256_b4x8_800k.py 的末尾
metrics = [
    dict(
        type='FrechetInceptionDistance',
        prefix='FID-Full-50k',
        fake_nums=50000,
        inception_style='StyleGAN',
        sample_model='ema'),
    dict(type='PrecisionAndRecall', fake_nums=50000, prefix='PR-50K'),
    dict(type='PerceptualPathLength', fake_nums=50000, prefix='ppl-w')
]
```

如上所述, `metrics` 由多个指标字典组成。 每个指标包含 `type` 来表示其类别。 `fake_nums` 表示模型生成的图像数量。
有些指标会输出一个结果字典，您也可以设置 `prefix` 来指定结果的前缀。
如果将FID的前缀设置为 `FID-Full-50k`，则输出的示例可能是

```bash
FID-Full-50k/fid: 3.6561  FID-Full-50k/mean: 0.4263  FID-Full-50k/cov: 3.2298
```

然后用户可以使用下面的命令测试模型:

```shell
bash tools/dist_test.sh ${CONFIG_FILE} ${CKPT_FILE}
```

如果您在 slurm 环境中，请使用如下命令切换到 [tools/slurm_test.sh](https://github.com/open-mmlab/mmagic/tree/main/tools/slurm_test.sh)：

```shell
sh slurm_test.sh ${PLATFORM} ${JOBNAME} ${CONFIG_FILE} ${CKPT_FILE}
```

## 在MMagic中训练模型

MMagic支持多种训练方式:

1. [在单个GPU上训练](#在单个GPU上训练)
2. [在单个GPU上训练](#在单个GPU上训练)
3. [在多个节点上训练](#在多个节点上训练)
4. [在Slurm上训练](#在Slurm上训练)

Specifically, all outputs (log files and checkpoints) will be saved to the working directory,
which is specified by `work_dir` in the config file.

### 在单个GPU上训练

```shell
CUDA_VISIBLE=0 python tools/train.py configs/example_config.py --work-dir work_dirs/example
```

### 在多个节点上训练

要在多台机器上启动分布式训练，这些机器可以通过IP访问，运行以下命令:

在第一台机器上:

```shell
NNODES=2 NODE_RANK=0 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR tools/dist_train.sh $CONFIG $GPUS
```

在第二台机器上:

```shell
NNODES=2 NODE_RANK=1 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR tools/dist_train.sh $CONFIG $GPUS
```

为了提高网络通信速度，建议使用高速网络硬件，如Infiniband。
请参考 [PyTorch docs](https://pytorch.org/docs/1.11/distributed.html#launch-utility) 以获取更多信息。

### 在多个GPU上训练

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

### 在Slurm上训练

如果您在由 [slurm](https://slurm.schedmd.com/) 管理的集群上运行MMagic，可以使用脚本`slurm_train.sh`。（此脚本还支持单机测试。）

```shell
[GPUS=${GPUS}] ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR}
```

下面是一个使用8个gpu在dev分区上训练inpainting模型的示例。

```shell
GPUS=8 ./tools/slurm_train.sh dev configs/inpainting/gl_places.py /nfs/xxxx/gl_places_256
```

你可以在 [slurm_train.sh](https://github.com/open-mmlab/mmagic/blob/master/tools/slurm_train.sh) 上查阅完整参数和环境变量。

### 可选参数

- `--amp`：此参数用于固定精度训练。
- `--resume`：此参数用于在训练中止时自动恢复。

## 使用特定的评估指标进行训练

受益于 `mmengine`的 `Runner`，我们可以在训练过程中对模型进行简单的评估，如下所示。

```python
# 定义指标
metrics = [
    dict(
        type='FrechetInceptionDistance',
        prefix='FID-Full-50k',
        fake_nums=50000,
        inception_style='StyleGAN')
]

# 定义dataloader
val_dataloader = dict(
    batch_size=128,
    num_workers=8,
    dataset=dict(
        type='BasicImageDataset',
        data_root='data/celeba-cropped/',
        pipeline=[
            dict(type='LoadImageFromFile', key='img'),
            dict(type='Resize', scale=(64, 64)),
            dict(type='PackInputs')
        ]),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True)

# 定义 val interval
train_cfg = dict(by_epoch=False, val_begin=1, val_interval=10000)

# 定义 val loop 和 evaluator
val_cfg = dict(type='MultiValLoop')
val_evaluator = dict(type='Evaluator', metrics=metrics)
```

可以设置 `val_begin` 和 `val_interval` 来调整何时开始验证和验证间隔。

有关指标的详细信息，请参考 [metrics' guide](./metrics.md).
