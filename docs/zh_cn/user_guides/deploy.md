# 教程 8：模型部署指南

[MMDeploy](https://github.com/open-mmlab/mmdeploy) 是 OpenMMLab 的部署仓库，负责包括 MMClassification、MMDetection、MMagic 等在内的各算法库的部署工作。
你可以从[这里](https://mmdeploy.readthedocs.io/zh_CN/latest/04-supported-codebases/mmagic.html)获取 MMDeploy 对 MMClassification 部署支持的最新文档。

本文的结构如下：

- [安装](#安装)
- [模型转换](#模型转换)
- [模型规范](#模型规范)
- [模型推理](#模型推理)
  - [后端模型推理](#后端模型推理)
  - [SDK 模型推理](#sdk-模型推理)
- [模型支持列表](#模型支持列表)

## 安装

请参考[此处](../get_started/install.md)安装 mmagic。然后，按照[说明](https://mmdeploy.readthedocs.io/zh_CN/latest/get_started.html#mmdeploy)安装 mmdeploy。

```{note}
如果安装的是 mmdeploy 预编译包，那么也请通过 'git clone https://github.com/open-mmlab/mmdeploy.git --depth=1' 下载 mmdeploy 源码。因为它包含了部署时要用到的配置文件
```

## 模型转换

假设在安装步骤中，mmagic 和 mmdeploy 代码库在同级目录下，并且当前的工作目录为 mmagic 的根目录，那么以 [ESRGAN](../../../configs/esrgan/esrgan_psnr-x4c64b23g32_1xb16-1000k_div2k.py) 模型为例，你可以从[此处](https://download.openmmlab.com/mmediting/restorers/esrgan/esrgan_psnr_x4c64b23g32_1x16_1000k_div2k_20200420-bf5c993c.pth)下载对应的 checkpoint，并使用以下代码将之转换为 onnx 模型：

```python
from mmdeploy.apis import torch2onnx
from mmdeploy.backend.sdk.export_info import export2SDK

img = 'tests/data/image/face/000001.png'
work_dir = 'mmdeploy_models/mmagic/onnx'
save_file = 'end2end.onnx'
deploy_cfg = '../mmdeploy/configs/mmagic/super-resolution/super-resolution_onnxruntime_dynamic.py'
model_cfg = 'configs/esrgan/esrgan_psnr-x4c64b23g32_1xb16-1000k_div2k.py'
model_checkpoint = 'esrgan_psnr_x4c64b23g32_1x16_1000k_div2k_20200420-bf5c993c.pth'
device = 'cpu'

# 1. convert model to onnx
torch2onnx(img, work_dir, save_file, deploy_cfg, model_cfg,
  model_checkpoint, device)

# 2. extract pipeline info for inference by MMDeploy SDK
export2SDK(deploy_cfg, model_cfg, work_dir, pth=model_checkpoint, device=device)
```

转换的关键之一是使用正确的配置文件。项目中已内置了各后端部署[配置文件](https://github.com/open-mmlab/mmdeploy/tree/main/configs/mmagic)。
文件的命名模式是：

```
{task}/{task}_{backend}-{precision}_{static | dynamic}_{shape}.py
```

其中：

- **{task}:** mmagic 中的任务
- **{backend}:** 推理后端名称。比如，onnxruntime、tensorrt、pplnn、ncnn、openvino、coreml 等等
- **{precision}:** 推理精度。比如，fp16、int8。不填表示 fp32
- **{static | dynamic}:** 动态、静态 shape
- **{shape}:** 模型输入的 shape 或者 shape 范围

在上例中，你也可以把 `ESRGAN` 转为其他后端模型。比如使用`super-resolution_tensorrt-fp16_dynamic-32x32-512x512.py`，把模型转为 tensorrt-fp16 模型。

```{tip}
当转 tensorrt 模型时, --device 需要被设置为 "cuda"
```

## 模型规范

在使用转换后的模型进行推理之前，有必要了解转换结果的结构。 它存放在 `--work-dir` 指定的路路径下。

上例中的`mmdeploy_models/mmagic/onnx`，结构如下：

```
mmdeploy_models/mmagic/onnx
├── deploy.json
├── detail.json
├── end2end.onnx
└── pipeline.json
```

重要的是：

- **end2end.onnx**: 推理引擎文件。可用 ONNX Runtime 推理
- ***xxx*.json**:  mmdeploy SDK 推理所需的 meta 信息

整个文件夹被定义为**mmdeploy SDK model**。换言之，**mmdeploy SDK model**既包括推理引擎，也包括推理 meta 信息。

## 模型推理

### 后端模型推理

以上述模型转换后的 `end2end.onnx` 为例，你可以使用如下代码进行推理：

```python
from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
import torch

deploy_cfg = '../mmdeploy/configs/mmagic/super-resolution/super-resolution_onnxruntime_dynamic.py'
model_cfg = 'configs/esrgan/esrgan_psnr-x4c64b23g32_1xb16-1000k_div2k.py'
device = 'cpu'
backend_model = ['mmdeploy_models/mmagic/onnx/end2end.onnx']
image = 'tests/data/image/lq/baboon_x4.png'

# read deploy_cfg and model_cfg
deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

# build task and backend model
task_processor = build_task_processor(model_cfg, deploy_cfg, device)
model = task_processor.build_backend_model(backend_model)

# process input image
input_shape = get_input_shape(deploy_cfg)
model_inputs, _ = task_processor.create_input(image, input_shape)

# do model inference
with torch.no_grad():
    result = model.test_step(model_inputs)

# visualize results
task_processor.visualize(
    image=image,
    model=model,
    result=result[0],
    window_name='visualize',
    output_file='output_restorer.bmp')
```

### SDK 模型推理

你也可以参考如下代码，对 SDK model 进行推理：

```python
from mmdeploy_python import Restorer
import cv2

img = cv2.imread('tests/data/image/lq/baboon_x4.png')

# create a predictor
restorer = Restorer(model_path='mmdeploy_models/mmagic/onnx', device_name='cpu', device_id=0)
# perform inference
result = restorer(img)

# visualize inference result
cv2.imwrite('output_restorer.bmp', result)
```

除了python API，mmdeploy SDK 还提供了诸如 C、C++、C#、Java等多语言接口。
你可以参考[样例](https://github.com/open-mmlab/mmdeploy/tree/main/demo)学习其他语言接口的使用方法。

## 模型支持列表

请参考[这里](https://mmdeploy.readthedocs.io/zh_CN/latest/04-supported-codebases/mmagic.html#id7)
