# ViCo (2023)

> [ViCo: Detail-Preserving Visual Condition for Personalized Text-to-Image Generation](https://arxiv.org/abs/2306.00971)

> **Task**: 文本图像生成

<!-- [ALGORITHM] -->

## 摘要

<!-- [ABSTRACT] -->

最近，个性化文本到图像生成使用扩散模型的方法被提出，并引起了广泛关注。给定包含新概念（例如独特的玩具）的少量图像，我们旨在调整生成模型，以捕捉新概念的精细视觉细节，并根据文本条件生成逼真的图像。我们提出了一种名为ViCo的插件方法，用于快速轻量级个性化生成。具体而言，我们提出了一个图像注意力模块，以对基于补丁的视觉语义进行扩散过程的条件建模。我们引入了一种基于注意力模块的对象蒙版，几乎没有额外计算成本。此外，我们设计了一个简单的正则化方法，基于文本-图像注意力图的内在属性，以减轻常见的过拟合退化问题。与许多现有模型不同，我们的方法不对原始扩散模型的任何参数进行微调。这使得模型的部署更加灵活和可转移。通过仅进行轻量级参数训练（约为扩散U-Net的6%），我们的方法在质量和数量上都达到了与所有最先进模型相当甚至更好的性能。

<!-- [IMAGE] -->

<div align=center>
<img src="https://github.com/haoosz/ViCo/assets/71176040/0ee95a57-fecf-4bba-bc64-eda46e5cc6d1">
</div>

## 模型结构

|       模型        |                                  数据集                                   | 下载 |
| :---------------: | :-----------------------------------------------------------------------: | :--: |
| [ViCo](./vico.py) | [textual_inversion_dataset](mmagic/datasets/textual_inversion_dataset.py) |  -   |

## Quick Start

1. 下载 [数据集](https://drive.google.com/drive/folders/1m8TCsY-C1tIOflHtWnFzTbw2C6dq67mC) 和 [模板](https://drive.google.com/drive/folders/1SpByLKECISmj5fhkaicT4yrsyqqpWL_T)
   and save to `data/vico/`

文件夹结构应如下:

```text
data
└── vico
    └──batman
       ├── 1.jpg
       ├── 2.jpg
       ├── 3.jpg
       └── 4.jpg
    └──clock
       ├── 1.jpg
       ├── 2.jpg
       ├── 3.jpg
       └── 4.jpg
    ...
    └──imagenet_templates_small.txt
```

2. 自定义你自己的config文件

```
# 请关注以下需自定义的内容

# 设置concept文件夹名
concept_dir = 'dog7'

# 设置代表这个concept的新字符
placeholder: str = 'S*'

# 初始化字符，最好是设置这个concept所属的类别
initialize_token: str = 'dog'
```

3. 使用以下命令进行**训练**:

```bash
# 4 GPUS
bash tools/dist_train.sh configs/vico/vico.py 4
# 1 GPU
python tools/train.py configs/vico/vico.py
```

4. 使用 [预训练的权重](https://drive.google.com/drive/folders/1GQGVzzOP2IgEfsQ-6ii6o2DqElnFThHM) 进行**推理**

```python
import torch
from mmengine import Config
from PIL import Image

from mmagic.registry import MODELS
from mmagic.utils import register_all_modules

register_all_modules()

# say you have downloaded the pretrained weights
cfg = Config.fromfile('configs/vico/dog.py')
state_dict = torch.load("./dog.pth")
vico = MODELS.build(cfg.model)
vico.load_state_dict(state_dict, strict=False)
vico = vico.cuda()

prompt = ["A photo of S*", "A photo of S* on the beach"]
reference = "data/vico/dog7/01.jpg"
image_ref = Image.open(reference)
with torch.no_grad():
    output = vico.infer(prompt=prompt, image_reference=image_ref, seed=123, num_images_per_prompt=2)['samples'][0]
output.save("infer.png")
```

5. (可选) 如果你想使用第3步训练得到的checkpoint进行推理，可以先使用以下脚本将训练过的参数提取出来（文件大小会轻量很多），再使用第4步进行推理

```python
import torch
def extract_vico_parameters(state_dict):
    new_state_dict = dict()
    for k, v in state_dict.items():
        if 'image_cross_attention' in k or 'trainable_embeddings' in k:
            new_k = k.replace('module.', '')
            new_state_dict[new_k] = v
    return new_state_dict

checkpoint = torch.load("work_dirs/vico/iter_400.pth")
new_checkpoint = extract_vico_parameters(checkpoint['state_dict'])
torch.save(new_checkpoint, "work_dirs/vico/dog.pth")
```

<table align="center">
<thead>
  <tr>
    <td>
<div align="center">
  <img src="https://github.com/open-mmlab/mmagic/assets/71176040/58a6953c-053a-40ea-8826-eee428c992b5" width="800"/>
  <br/>
  <b>'vico'</b>
</thead>
</table>

## Comments

Our codebase for the stable diffusion models builds heavily on [diffusers codebase](https://github.com/huggingface/diffusers) and the model weights are from [stable-diffusion-1.5](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_controlnet.py).

Thanks for the efforts of the community!

## Citation

```bibtex
@inproceedings{Hao2023ViCo,
  title={ViCo: Detail-Preserving Visual Condition for Personalized Text-to-Image Generation},
  author={Shaozhe Hao and Kai Han and Shihao Zhao and Kwan-Yee K. Wong},
  year={2023}
}
```
