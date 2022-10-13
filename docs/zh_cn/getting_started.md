# 入门指引

这里我们将提供关于如何使用 MMEditing 的基础教程。这里我们默认你已经安装好 MMEditing，安装教程可以参考 [install.md](install.md)。

## 准备数据集

推荐将数据集根目录链接到 $MMEditing/data 文件夹。
如果你的文件夹结构不是如此，你可能需要修改一下配置文件中与之相关的数据路径。
下面是各种任务中，我们需要用到的数据集的介绍：

[Inpainting](https://mmediting.readthedocs.io/en/latest/_tmp/inpainting_datasets.html)

[Matting](https://mmediting.readthedocs.io/en/latest/_tmp/matting_datasets.html)

[Restoration](https://mmediting.readthedocs.io/en/latest/_tmp/sr_datasets.html)

[Generation](https://mmediting.readthedocs.io/en/latest/_tmp/generation_datasets.html)

## 用预训练模型做推理

我们提供测试脚本来评估整个数据集，以及一些针对特定任务的运行样例。

### 测试一个数据集

MMEditing 基于 `MMDistributedDataParallel` 实现了分布式测试。

#### 用单/多 GPU 进行测试

你可以按照下面的命令使用单个或者多个 GPU 进行测试：

```shell
# 单GPU测试
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--save-path ${IMAGE_SAVE_PATH}]

# 多GPU测试
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--save-path ${IMAGE_SAVE_PATH}]
```

具体来说，比如：

```shell
# 单GPU测试
python tools/test.py configs/example_config.py work_dirs/example_exp/example_model_20200202.pth --out work_dirs/example_exp/results.pkl

# 多GPU测试
./tools/dist_test.sh configs/example_config.py work_dirs/example_exp/example_model_20200202.pth 2 --save-path work_dirs/example_exp/results/
```

#### 使用 Slurm 系统进行测试

如果你在一个部署了 slurm 系统的集群上进行测试，你可以使用 `slurm_test.sh` 进行测试。（这个脚本也支持单卡测试。）

```shell
[GPUS=${GPUS}] ./tools/slurm_test.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${CHECKPOINT_FILE}
```

这里有一个使用8个 GPU 进行测试的样例，我们使用 `dev` 分区，同时设置任务名字为 `test`：

```shell
GPUS=8 ./tools/slurm_test.sh dev test configs/example_config.py work_dirs/example_exp/example_model_20200202.pth
```

你可以查看 [slurm_test.sh](https://github.com/open-mmlab/mmediting/blob/master/tools/slurm_test.sh) 脚本来获得所有的参数和环境变量。

#### 可选参数

- `--out`: 指定 pickle 格式的输出结果的文件名。如果没有被指定，结果将不会被保存到某个文件里面。
- `--save-path`: 指定一个路径来存储编辑过的图片。如果没有被指定，图片将不会被存储。
- `--seed`: 测试过程中的随机种子。这个参数是用来固定输出结果的。
- `--deterministic`: 与 `--seed` 相关，这个参数决定了是否设置对于 CUDNN 的决定性选项。如果被指定了，`torch.backends.cudnn.deterministic` 将被设为 True，并且 `torch.backends.cudnn.benchmark` 将被设为 False。
- `--cfg-options`:  如果指明，这里的键值对将会被合并到配置文件中。

注：当前，我们不支持像 MMDetection 一样用 --eval 参数来指定评测指标。在 MMEditing 中，我们在配置文件中指定评测指标（详情参考：[config.md](config.md)）。

### 图像样例

我们提供了一些特定任务的测试样例。

#### Inpainting

你可以使用下面的命令来测试图片以及对应的 mask。

```shell
python demo/inpainting_demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${MASKED_IMAGE_FILE} ${MASK_FILE} ${SAVE_FILE} [--imshow] [--device ${GPU_ID}]
```

如果 `--imshow` 被指定，这个样例也能通过 opencv 展示图片。比如：

```shell
python demo/inpainting_demo.py configs/inpainting/global_local/gl_256x256_8x12_celeba.py xxx.pth tests/data/image/celeba_test.png tests/data/image/bbox_mask.png tests/data/pred/inpainting_celeba.png
```

预测结果将被存储在 `tests/data/pred/inpainting_celeba.png`。

#### Matting

你可以使用下面的命令来测试图片以及对应的三元组（trimap）。

```shell
python demo/matting_demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${IMAGE_FILE} ${TRIMAP_FILE} ${SAVE_FILE} [--imshow] [--device ${GPU_ID}]
```

如果 `--imshow` 被指定，这个样例也能通过 opencv 展示图片。比如：

```shell
python demo/matting_demo.py configs/mattors/dim/dim_stage3_v16_pln_1x1_1000k_comp1k.py work_dirs/dim_stage3/latest.pth tests/data/merged/GT05.jpg tests/data/trimap/GT05.png tests/data/pred/GT05.png
```

预测的 alpha 图将被保存到 `tests/data/pred/GT05.png`。

#### Restoration

你可以使用如下命令来恢复图片：

```shell
python demo/restoration_demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${IMAGE_FILE} ${SAVE_FILE} [--imshow] [--device ${GPU_ID}]
```

如果 `--imshow` 被指定，这个样例也能通过 opencv 展示图片。比如：

```shell
python demo/restoration_demo.py configs/restorers/esrgan/esrgan_x4c64b23g32_1x16_400k_div2k.py work_dirs/esrgan_x4c64b23g32_1x16_400k_div2k/latest.pth tests/data/lq/baboon_x4.png demo/demo_out_baboon.png
```

恢复的图片将被存储在 `demo/demo_out_baboon.png`。

#### Generation

```shell
python demo/generation_demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${IMAGE_FILE} ${SAVE_FILE} [--unpaired-path ${UNPAIRED_IMAGE_FILE}] [--imshow] [--device ${GPU_ID}]
```

如果 `--unpaired-path` 被指定了（适用于 CycleGAN），模型将会进行图片到图片的变换。如果 `--imshow` 被指定，这个脚本也能通过 opencv 展示图片。比如：

成对的模型:

```shell
python demo/generation_demo.py configs/example_config.py work_dirs/example_exp/example_model_20200202.pth demo/demo.jpg demo/demo_out.jpg
```

非成对的模型 (使用 opencv 展示图片):

```shell
python demo/generation_demo.py configs/example_config.py work_dirs/example_exp/example_model_20200202.pth demo/demo.jpg demo/demo_out.jpg --unpaired-path demo/demo_unpaired.jpg --imshow
```

## 训练一个模型

MMEditing 基于 `MMDistributedDataParallel` 实现了分布式训练。

所有的输出（日志文件和模型权重文件）将会被存储到工作目录下面，工作目录可以通过 `work_dir` 参数在配置文件中指定。

我们默认地会在训练几个 iteration 之后在验证集测试自己的模型，你可以在训练配置文件中设置测试间隔：

```python
evaluation = dict(interval=1e4, by_epoch=False)  # 这样的话模型就会每 10,000 次迭代就进行评测
```

### 用单/多 GPUs 训练模型

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

可选参数是:

- `--no-validate` (**不建议采用**): 我们默认会每 k 步就进行评测。如果需要关闭这个功能，可以使用 `--no-validate` 参数。
- `--work-dir ${WORK_DIR}`: 重写在配置文件中指定的工作目录。
- `--resume-from ${CHECKPOINT_FILE}`: 从之前的模型权重文件中重启训练。
- `--cfg-options`:  如果指明，这里的键值对将会被合并到配置文件中。

`resume-from` 和 `load-from` 之间的区别:
`resume-from` 会加载模型参数、优化的状态量以及特定 checkpoint 里面的训练 iteration 数。这个参数通常会被用来重启被意外打断的训练过程。
`load-from` 只会加载模型的参数，并且训练的 iteration 会从 0 开始。这个模式通常会用来微调模型。

### 用 Slurm 进行训练

如果你在用 [slurm](https://slurm.schedmd.com/) 集群跑 MMEditing，你可以用 `slurm_train.sh` 这个脚本。（这个脚本也支持单机训练）

```shell
[GPUS=${GPUS}] ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR}
```

这里是一个使用 8 GPUs 训练的例子：

```shell
GPUS=8 ./tools/slurm_train.sh dev places_256 configs/inpainting/gl_places.py /nfs/xxxx/gl_places_256
```

你可以查看 [slurm_train.sh](https://github.com/open-mmlab/mmediting/blob/master/tools/slurm_train.sh) 来得到所有的参数和环境变量。

### 启动多个任务在单个机器上

如果你启动多个任务在单个机器上，比如：2个 4-GPU 任务在一个8 GPUs 机器上。你需要为每一个任务指定不同的端口来避免信息交互的冲突。

如果你是用 `dist_train.sh` 来启动训练的任务，你可以用如下命令设置端口：

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh ${CONFIG_FILE} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh ${CONFIG_FILE} 4
```

如果你在 slurm 系统上启动训练任务，你需要修改配置文件来设定不同的信息交互端口：

在 `config1.py` 中：

```python
dist_params = dict(backend='nccl', port=29500)
```

在 `config2.py` 中,

```python
dist_params = dict(backend='nccl', port=29501)
```

接下来你可以使用 config1.py 和 config2.py 启动两个任务：

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py ${WORK_DIR}
CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py ${WORK_DIR}
```

## 有用的工具

我们提供许多有用的工具，他们放到了 `tools/` 文件夹下。

### 得到 FLOPs 和 参数数量

我们提供一个基于 [flops-counter.pytorch](https://github.com/sovrasov/flops-counter.pytorch) 修改的脚本来计算 FLOPs 和 一个指定模型的参数。

```shell
python tools/get_flops.py ${CONFIG_FILE} [--shape ${INPUT_SHAPE}]
```

具体来说：

```shell
python tools/get_flops.py configs/resotorer/srresnet.py --shape 40 40
```

你将会得到如下输出：

```
==============================
Input shape: (3, 40, 40)
Flops: 4.07 GMac
Params: 1.52 M
==============================
```

**注意**：这个工具目前还是出试验阶段，我们不能确保这个数值是正确的。当你需要在技术报告或者论文中使用时，请谨慎查看这个数值：

(1) FLOPs 跟输入的形状非常相关。默认的输入形状是 (1, 3, 250, 250)。
(2) 一些算子没有被算入 FLOPs 里面，比如像 GN 和一些自定义的算子。
你能够通过修改 [`mmcv/cnn/utils/flops_counter.py`](https://github.com/open-mmlab/mmcv/blob/master/mmcv/cnn/utils/flops_counter.py) 来增加对新算子的支持。

### 发布一个模型

在你上传模型到 AWS 之前，你需要：
(1) 将模型的权重转换为 CPU 向量, (2) 删除优化器的状态量，
(3) 计算 checkpoint 的哈希码，并且将哈希码添加到文件名后面。

```shell
python tools/publish_model.py ${INPUT_FILENAME} ${OUTPUT_FILENAME}
```

具体来说：

```shell
python tools/publish_model.py work_dirs/example_exp/latest.pth example_model_20200202.pth
```

最终的输出文件名将会是 `example_model_20200202-{hash id}.pth`。
