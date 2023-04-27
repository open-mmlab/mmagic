# Tutorial 8: Deploy models in MMagic

The deployment of OpenMMLab codebases, including MMClassification, MMDetection, MMagic and so on are supported by [MMDeploy](https://github.com/open-mmlab/mmdeploy).
The latest deployment guide for MMagic can be found from [here](https://mmdeploy.readthedocs.io/en/latest/04-supported-codebases/mmagic.html).

This tutorial is organized as follows:

- [Tutorial 8: Deploy models in MMagic](#tutorial-8-deploy-models-in-mmagic)
  - [Installation](#installation)
  - [Convert model](#convert-model)
  - [Model specification](#model-specification)
  - [Model inference](#model-inference)
    - [Backend model inference](#backend-model-inference)
    - [SDK model inference](#sdk-model-inference)
  - [Supported models](#supported-models)

## Installation

Please follow the [guide](../get_started/install.md) to install mmagic. And then install mmdeploy from source by following [this](https://mmdeploy.readthedocs.io/en/latest/get_started.html#installation) guide.

```{note}
If you install mmdeploy prebuilt package, please also clone its repository by 'git clone https://github.com/open-mmlab/mmdeploy.git --depth=1' to get the deployment config files.
```

## Convert model

Suppose MMagic and mmdeploy repositories are in the same directory, and the working directory is the root path of MMagic.

Take [ESRGAN](../../../configs/esrgan/esrgan_psnr-x4c64b23g32_1xb16-1000k_div2k.py) model as an example.
You can download its checkpoint from [here](https://download.openmmlab.com/MMagic/restorers/esrgan/esrgan_psnr_x4c64b23g32_1x16_1000k_div2k_20200420-bf5c993c.pth), and then convert it to onnx model as follows:

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

It is crucial to specify the correct deployment config during model conversion.MMDeploy has already provided builtin deployment config [files](https://github.com/open-mmlab/mmdeploy/tree/main/configs/mmagic) of all supported backends for mmagic, under which the config file path follows the pattern:

```
{task}/{task}_{backend}-{precision}_{static | dynamic}_{shape}.py
```

- **{task}:** task in mmagic.

- **{backend}:** inference backend, such as onnxruntime, tensorrt, pplnn, ncnn, openvino, coreml etc.

- **{precision}:** fp16, int8. When it's empty, it means fp32

- **{static | dynamic}:** static shape or dynamic shape

- **{shape}:** input shape or shape range of a model

Therefore, in the above example, you can also convert `ESRGAN` to other backend models by changing the deployment config file, e.g., converting to tensorrt-fp16 model by `super-resolution_tensorrt-fp16_dynamic-32x32-512x512.py`.

```{tip}
When converting mmagic models to tensorrt models, --device should be set to "cuda"
```

## Model specification

Before moving on to model inference chapter, let's know more about the converted model structure which is very important for model inference.

The converted model locates in the working directory like `mmdeploy_models/mmagic/onnx` in the previous example. It includes:

```
mmdeploy_models/mmagic/onnx
├── deploy.json
├── detail.json
├── end2end.onnx
└── pipeline.json
```

in which,

- **end2end.onnx**: backend model which can be inferred by ONNX Runtime
- ***xxx*.json**: the necessary information for mmdeploy SDK

The whole package **mmdeploy_models/mmagic/onnx** is defined as **mmdeploy SDK model**, i.e., **mmdeploy SDK model** includes both backend model and inference meta information.

## Model inference

### Backend model inference

Take the previous converted `end2end.onnx` model as an example, you can use the following code to inference the model.

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

### SDK model inference

You can also perform SDK model inference like following,

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

Besides python API, MMDeploy SDK also provides other FFI (Foreign Function Interface), such as C, C++, C#, Java and so on. You can learn their usage from [demos](https://github.com/open-mmlab/mmdeploy/tree/main/demo).

## Supported models

Please refer to [here](https://mmdeploy.readthedocs.io/en/latest/04-supported-codebases/mmagic.html#supported-models) for the supported model list.
