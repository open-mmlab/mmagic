## 使用预训练模型进行推理

我们提供用于在完整数据集上进行预训练模型评估和特定任务图像演示的测试脚本。

### 测试一个预训练模型

MMEditing 使用 `MMDistributedDataParallel` 实现 **分布式**测试。

#### 在单/多个 GPU 上进行测试

您可以使用以下命令在单/多个 GPU 上测试预训练模型。

```shell
# 单 GPU 测试
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--save-path ${IMAGE_SAVE_PATH}]

# 多 GPU 测试
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--save-path ${IMAGE_SAVE_PATH}]
```

例如

```shell
# 单 GPU 测试
python tools/test.py configs/example_config.py work_dirs/example_exp/example_model_20200202.pth --out work_dirs/example_exp/results.pkl

# 多 GPU 测试
./tools/dist_test.sh configs/example_config.py work_dirs/example_exp/example_model_20200202.pth --save-path work_dirs/example_exp/results/
```

#### 在 slurm 上测试

如果您在使用 [slurm](https://slurm.schedmd.com/) 管理的集群上运行 MMEditing，则可以使用脚本 `slurm_test.sh`。（此脚本也支持单机测试。）

```shell
[GPUS=${GPUS}] ./tools/slurm_test.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${CHECKPOINT_FILE}
```

以下是使用 8 个 GPU 在作业名称为 `test` 的 `dev` 分区上测试示例模型的例子。

```shell
GPUS=8 ./tools/slurm_test.sh dev test configs/example_config.py work_dirs/example_exp/example_model_20200202.pth
```

您可以查看 [slurm_test.sh](https://github.com/open-mmlab/mmediting/blob/master/tools/slurm_test.sh) 以获取完整的参数和环境变量。

#### 可选参数

- `--out`: 以 pickle 格式指定输出结果的文件名。 如果没有给出，结果将不会保存到文件中。
- `--save-path`: 指定存储编辑图像的路径。 如果没有给出，图像将不会被保存。
- `--seed`: 测试期间的随机种子。 此参数用于固定某些任务中的结果，例如*修复*。
- `--deterministic`: 与 `--seed` 相关，此参数决定是否为 CUDNN 后端设置确定性的选项。如果指定该参数，会将 `torch.backends.cudnn.deterministic` 设置为 `True`，将 `torch.backends.cudnn.benchmark` 设置为 `False`。
- `--cfg-options`:  如果指明，这里的键值对将会被合并到配置文件中。

注：目前，我们不使用像 [MMDetection](https://github.com/open-mmlab/mmdetection) 那样的 `--eval` 参数来指定评估指标。 评估指标在配置文件中给出（参见 [config.md](config.md)）。

## 训练一个模型

MMEditing 使用 `MMDistributedDataParallel` 实现 **分布式**测试。

所有输出（日志文件和模型权重文件）都将保存到工作目录中，
工作目录由配置文件中的 `work_dir` 指定。

默认情况下，我们在多次迭代后评估验证集上的模型，您可以通过在训练配置中添加 `interval` 参数来更改评估间隔。

```python
evaluation = dict(interval=1e4, by_epoch=False)  # 每一万次迭代进行一次评估。
```

### 在单/多个 GPU 上训练

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

可选参数是：

- `--no-validate` (**不建议**): 默认情况下，代码库将在训练期间每 k 次迭代执行一次评估。若要禁用此行为，请使用 `--no-validate`。
- `--work-dir ${WORK_DIR}`: 覆盖配置文件中指定的工作目录。
- `--resume-from ${CHECKPOINT_FILE}`: 从已有的模型权重文件恢复。
- `--cfg-options`:  如果指明，这里的键值对将会被合并到配置文件中。

`resume-from` 和 `load-from` 之间的区别：
`resume-from` 加载模型权重和优化器状态，迭代也从指定的检查点继承。 它通常用于恢复意外中断的训练过程。
`load-from` 只加载模型权重，训练迭代从 0 开始，通常用于微调。

#### 使用多节点训练

如果您有多个计算节点，而且他们可以通过 IP 互相访问，可以使用以下命令启动分布式训练：

在第一个节点：

```shell
NNODES=2 NODE_RANK=0 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR tools/dist_train.sh $CONFIG $GPUS
```

在第二个节点：

```shell
NNODES=2 NODE_RANK=1 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR tools/dist_train.sh $CONFIG $GPUS
```

为提高网络通信速度，推荐使用高速网络设备，如 Infiniband 等。
更多信息可参照[PyTorch 文档](https://pytorch.org/docs/1.11/distributed.html#launch-utility).

### 在 slurm 上训练

如果您在使用 [slurm](https://slurm.schedmd.com/) 管理的集群上运行 MMEditing，则可以使用脚本 `slurm_train.sh`。（此脚本也支持单机训练。）

```shell
[GPUS=${GPUS}] ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR}
```

以下是使用 8 个 GPU 在 `dev` 分区上训练*修复*模型的示例。

```shell
GPUS=8 ./tools/slurm_train.sh dev configs/inpainting/gl_places.py /nfs/xxxx/gl_places_256
```

您可以查看 [slurm_train.sh](https://github.com/open-mmlab/mmediting/blob/master/tools/slurm_train.sh) 以获取完整的参数和环境变量。

### 在一台机器上启动多个作业

如果您在一台机器上启动多个作业，例如，在具有 8 个 GPU 的机器上进行 2 个 4-GPU 训练的作业，
您需要为每个作业指定不同的端口（默认为 29500）以避免通信冲突。

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh ${CONFIG_FILE} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh ${CONFIG_FILE} 4
```

如果您使用 Slurm 启动训练作业，则需要修改配置文件（通常是配置文件的倒数第 6 行）以设置不同的通信端口。

在 `config1.py` 中,

```python
dist_params = dict(backend='nccl', port=29500)
```

在 `config2.py` 中,

```python
dist_params = dict(backend='nccl', port=29501)
```

然后您可以使用 `config1.py` 和 `config2.py` 启动两个作业。

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py ${WORK_DIR}
CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py ${WORK_DIR}
```
