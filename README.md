<div id="top" align="center">
  <img src="docs/en/_static/image/mmagic-logo.png" width="500px"/>
  <div>&nbsp;</div>
  <div align="center">
    <font size="10"><b>M</b>ultimodal <b>A</b>dvanced, <b>G</b>enerative, and <b>I</b>ntelligent <b>C</b>reation (MMagic [em'mædʒɪk])</font>
  </div>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab website</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab platform</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

[![PyPI](https://badge.fury.io/py/mmagic.svg)](https://pypi.org/project/mmagic/)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmagic.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmagic/workflows/build/badge.svg)](https://github.com/open-mmlab/mmagic/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmagic/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmagic)
[![license](https://img.shields.io/github/license/open-mmlab/mmagic.svg)](https://github.com/open-mmlab/mmagic/blob/main/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmagic.svg)](https://github.com/open-mmlab/mmagic/issues)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmagic.svg)](https://github.com/open-mmlab/mmagic/issues)
[![Open in OpenXLab](https://cdn-static.openxlab.org.cn/app-center/openxlab_demo.svg)](https://openxlab.org.cn/apps/detail/%E6%94%BF%E6%9D%B0/OpenMMLab-Projects)

[📘Documentation](https://mmagic.readthedocs.io/en/latest/) |
[🛠️Installation](https://mmagic.readthedocs.io/en/latest/get_started/install.html) |
[📊Model Zoo](https://mmagic.readthedocs.io/en/latest/model_zoo/overview.html) |
[🆕Update News](https://mmagic.readthedocs.io/en/latest/changelog.html) |
[🚀Ongoing Projects](https://github.com/open-mmlab/mmagic/projects) |
[🤔Reporting Issues](https://github.com/open-mmlab/mmagic/issues)

English | [简体中文](README_zh-CN.md)

</div>

<div align="center">
  <a href="https://openmmlab.medium.com/" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218352562-cdded397-b0f3-4ca1-b8dd-a60df8dca75b.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://discord.gg/raweFPmdzG" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218347213-c080267f-cbb6-443e-8532-8e1ed9a58ea9.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://twitter.com/OpenMMLab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346637-d30c8a0f-3eba-4699-8131-512fb06d46db.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://www.youtube.com/openmmlab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346691-ceb2116a-465a-40af-8424-9f30d2348ca9.png" width="3%" alt="" /></a>
</div>

## 🚀 What's New <a><img width="35" height="20" src="https://user-images.githubusercontent.com/12782558/212848161-5e783dd6-11e8-4fe0-bbba-39ffb77730be.png"></a>

### New release [**MMagic v1.2.0**](https://github.com/open-mmlab/mmagic/releases/tag/v1.2.0) \[18/12/2023\]:

- An advanced and powerful inpainting algorithm named PowerPaint is released in our repository. [Click to View](https://github.com/open-mmlab/mmagic/tree/main/projects/powerpaint)

We are excited to announce the release of MMagic v1.0.0 that inherits from [MMEditing](https://github.com/open-mmlab/mmediting) and [MMGeneration](https://github.com/open-mmlab/mmgeneration).

After iterative updates with OpenMMLab 2.0 framework and merged with MMGeneration, MMEditing has become a powerful tool that supports low-level algorithms based on both GAN and CNN. Today, MMEditing embraces Generative AI and transforms into a more advanced and comprehensive AIGC toolkit: **MMagic** (**M**ultimodal **A**dvanced, **G**enerative, and **I**ntelligent **C**reation). MMagic will provide more agile and flexible experimental support for researchers and AIGC enthusiasts, and help you on your AIGC exploration journey.

We highlight the following new features.

**1. New Models**

We support 11 new models in 4 new tasks.

- Text2Image / Diffusion
  - ControlNet
  - DreamBooth
  - Stable Diffusion
  - Disco Diffusion
  - GLIDE
  - Guided Diffusion
- 3D-aware Generation
  - EG3D
- Image Restoration
  - NAFNet
  - Restormer
  - SwinIR
- Image Colorization
  - InstColorization

**2. Magic Diffusion Model**

For the Diffusion Model, we provide the following "magic" :

- Support image generation based on Stable Diffusion and Disco Diffusion.
- Support Finetune methods such as Dreambooth and DreamBooth LoRA.
- Support controllability in text-to-image generation using ControlNet.
- Support acceleration and optimization strategies based on xFormers to improve training and inference efficiency.
- Support video generation based on MultiFrame Render.
- Support calling basic models and sampling strategies through DiffuserWrapper.

**3. Upgraded Framework**

By using MMEngine and MMCV of OpenMMLab 2.0 framework, MMagic has upgraded in the following new features:

- Refactor DataSample to support the combination and splitting of batch dimensions.
- Refactor DataPreprocessor and unify the data format for various tasks during training and inference.
- Refactor MultiValLoop and MultiTestLoop, supporting the evaluation of both generation-type metrics (e.g. FID) and reconstruction-type metrics (e.g. SSIM), and supporting the evaluation of multiple datasets at once.
- Support visualization on local files or using tensorboard and wandb.
- Support for 33+ algorithms accelerated by Pytorch 2.0.

**MMagic** has supported all the tasks, models, metrics, and losses in [MMEditing](https://github.com/open-mmlab/mmediting) and [MMGeneration](https://github.com/open-mmlab/mmgeneration) and unifies interfaces of all components based on [MMEngine](https://github.com/open-mmlab/mmengine) 😍.

Please refer to [changelog.md](docs/en/changelog.md) for details and release history.

Please refer to [migration documents](docs/en/migration/overview.md) to migrate from [old version](https://github.com/open-mmlab/mmagic/tree/0.x) MMEditing 0.x to new version MMagic 1.x .

<div id="table" align="center"></div>

## 📄 Table of Contents

- [📖 Introduction](#-introduction)
- [🙌 Contributing](#-contributing)
- [🛠️ Installation](#️-installation)
- [📊 Model Zoo](#-model-zoo)
- [🤝 Acknowledgement](#-acknowledgement)
- [🖊️ Citation](#️-citation)
- [🎫 License](#-license)
- [🏗️ ️OpenMMLab Family](#️-️openmmlab-family)

## 📖 Introduction

MMagic (**M**ultimodal **A**dvanced, **G**enerative, and **I**ntelligent **C**reation) is an advanced and comprehensive AIGC toolkit that inherits from [MMEditing](https://github.com/open-mmlab/mmediting) and [MMGeneration](https://github.com/open-mmlab/mmgeneration). It is an open-source image and video editing&generating toolbox based on PyTorch. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

Currently, MMagic support multiple image and video generation/editing tasks.

https://user-images.githubusercontent.com/49083766/233564593-7d3d48ed-e843-4432-b610-35e3d257765c.mp4

### ✨ Major features

- **State of the Art Models**

  MMagic provides state-of-the-art generative models to process, edit and synthesize images and videos.

- **Powerful and Popular Applications**

  MMagic supports popular and contemporary image restoration, text-to-image, 3D-aware generation, inpainting, matting, super-resolution and generation applications. Specifically, MMagic supports fine-tuning for stable diffusion and many exciting diffusion's application such as ControlNet Animation with SAM. MMagic also supports GAN interpolation, GAN projection, GAN manipulations and many other popular GAN’s applications. It’s time to begin your AIGC exploration journey!

- **Efficient Framework**

  By using MMEngine and MMCV of OpenMMLab 2.0 framework, MMagic decompose the editing framework into different modules and one can easily construct a customized editor framework by combining different modules. We can define the training process just like playing with Legos and provide rich components and strategies. In MMagic, you can complete controls on the training process with different levels of APIs. With the support of [MMSeparateDistributedDataParallel](https://github.com/open-mmlab/mmengine/blob/main/mmengine/model/wrappers/seperate_distributed.py), distributed training for dynamic architectures can be easily implemented.

### ✨ Best Practice

- The best practice on our main branch works with **Python 3.9+** and **PyTorch 2.0+**.

<p align="right"><a href="#table">🔝Back to Table of Contents</a></p>

## 🙌 Contributing

More and more community contributors are joining us to make our repo better. Some recent projects are contributed by the community including:

- [SDXL](configs/stable_diffusion_xl/README.md) is contributed by  @okotaku.
- [AnimateDiff](configs/animatediff/README.md) is contributed by @ElliotQi.
- [ViCo](configs/vico/README.md) is contributed by @FerryHuang.
- [DragGan](configs/draggan/README.md) is contributed by @qsun1.
- [FastComposer](configs/fastcomposer/README.md) is contributed by @xiaomile.

[Projects](projects/README.md) is opened to make it easier for everyone to add projects to MMagic.

We appreciate all contributions to improve MMagic. Please refer to [CONTRIBUTING.md](https://github.com/open-mmlab/mmcv/blob/main/CONTRIBUTING.md) in MMCV and [CONTRIBUTING.md](https://github.com/open-mmlab/mmengine/blob/main/CONTRIBUTING.md) in MMEngine for more details about the contributing guideline.

<p align="right"><a href="#table">🔝Back to Table of Contents</a></p>

## 🛠️ Installation

MMagic depends on [PyTorch](https://pytorch.org/), [MMEngine](https://github.com/open-mmlab/mmengine) and [MMCV](https://github.com/open-mmlab/mmcv).
Below are quick steps for installation.

**Step 1.**
Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/).

**Step 2.**
Install MMCV, MMEngine and MMagic with [MIM](https://github.com/open-mmlab/mim).

```shell
pip3 install openmim
mim install mmcv>=2.0.0
mim install mmengine
mim install mmagic
```

**Step 3.**
Verify MMagic has been successfully installed.

```shell
cd ~
python -c "import mmagic; print(mmagic.__version__)"
# Example output: 1.0.0
```

**Getting Started**

After installing MMagic successfully, now you are able to play with MMagic! To generate an image from text, you only need several lines of codes by MMagic!

```python
from mmagic.apis import MMagicInferencer
sd_inferencer = MMagicInferencer(model_name='stable_diffusion')
text_prompts = 'A panda is having dinner at KFC'
result_out_dir = 'output/sd_res.png'
sd_inferencer.infer(text=text_prompts, result_out_dir=result_out_dir)
```

Please see [quick run](docs/en/get_started/quick_run.md) and [inference](docs/en/user_guides/inference.md) for the basic usage of MMagic.

**Install MMagic from source**

You can also experiment on the latest developed version rather than the stable release by installing MMagic from source with the following commands:

```shell
git clone https://github.com/open-mmlab/mmagic.git
cd mmagic
pip3 install -e .
```

Please refer to [installation](docs/en/get_started/install.md) for more detailed instruction.

<p align="right"><a href="#table">🔝Back to Table of Contents</a></p>

## 📊 Model Zoo

<div align="center">
  <b>Supported algorithms</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Conditional GANs</b>
      </td>
      <td>
        <b>Unconditional GANs</b>
      </td>
      <td>
        <b>Image Restoration</b>
      </td>
      <td>
        <b>Image Super-Resolution</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
            <li><a href="configs/sngan_proj/README.md">SNGAN/Projection GAN (ICLR'2018)</a></li>
            <li><a href="configs/sagan/README.md">SAGAN (ICML'2019)</a></li>
            <li><a href="configs/biggan/README.md">BIGGAN/BIGGAN-DEEP (ICLR'2018)</a></li>
      </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/dcgan/README.md">DCGAN (ICLR'2016)</a></li>
          <li><a href="configs/wgan-gp/README.md">WGAN-GP (NeurIPS'2017)</a></li>
          <li><a href="configs/lsgan/README.md">LSGAN (ICCV'2017)</a></li>
          <li><a href="configs/ggan/README.md">GGAN (ArXiv'2017)</a></li>
          <li><a href="configs/pggan/README.md">PGGAN (ICLR'2018)</a></li>
          <li><a href="configs/singan/README.md">SinGAN (ICCV'2019)</a></li>
          <li><a href="configs/styleganv1/README.md">StyleGANV1 (CVPR'2019)</a></li>
          <li><a href="configs/styleganv2/README.md">StyleGANV2 (CVPR'2019)</a></li>
          <li><a href="configs/styleganv3/README.md">StyleGANV3 (NeurIPS'2021)</a></li>
          <li><a href="configs/draggan/README.md">DragGan (2023)</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/swinir/README.md">SwinIR (ICCVW'2021)</a></li>
          <li><a href="configs/nafnet/README.md">NAFNet (ECCV'2022)</a></li>
          <li><a href="configs/restormer/README.md">Restormer (CVPR'2022)</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/srcnn/README.md">SRCNN (TPAMI'2015)</a></li>
          <li><a href="configs/srgan_resnet/README.md">SRResNet&SRGAN (CVPR'2016)</a></li>
          <li><a href="configs/edsr/README.md">EDSR (CVPR'2017)</a></li>
          <li><a href="configs/esrgan/README.md">ESRGAN (ECCV'2018)</a></li>
          <li><a href="configs/rdn/README.md">RDN (CVPR'2018)</a></li>
          <li><a href="configs/dic/README.md">DIC (CVPR'2020)</a></li>
          <li><a href="configs/ttsr/README.md">TTSR (CVPR'2020)</a></li>
          <li><a href="configs/glean/README.md">GLEAN (CVPR'2021)</a></li>
          <li><a href="configs/liif/README.md">LIIF (CVPR'2021)</a></li>
          <li><a href="configs/real_esrgan/README.md">Real-ESRGAN (ICCVW'2021)</a></li>
        </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
<tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Video Super-Resolution</b>
      </td>
      <td>
        <b>Video Interpolation</b>
      </td>
      <td>
        <b>Image Colorization</b>
      </td>
      <td>
        <b>Image Translation</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
            <li><a href="configs/edvr/README.md">EDVR (CVPR'2018)</a></li>
            <li><a href="configs/tof/README.md">TOF (IJCV'2019)</a></li>
            <li><a href="configs/tdan/README.md">TDAN (CVPR'2020)</a></li>
            <li><a href="configs/basicvsr/README.md">BasicVSR (CVPR'2021)</a></li>
            <li><a href="configs/iconvsr/README.md">IconVSR (CVPR'2021)</a></li>
            <li><a href="configs/basicvsr_pp/README.md">BasicVSR++ (CVPR'2022)</a></li>
            <li><a href="configs/real_basicvsr/README.md">RealBasicVSR (CVPR'2022)</a></li>
      </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/tof/README.md">TOFlow (IJCV'2019)</a></li>
          <li><a href="configs/cain/README.md">CAIN (AAAI'2020)</a></li>
          <li><a href="configs/flavr/README.md">FLAVR (CVPR'2021)</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/inst_colorization/README.md">InstColorization (CVPR'2020)</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/pix2pix/README.md">Pix2Pix (CVPR'2017)</a></li>
          <li><a href="configs/cyclegan/README.md">CycleGAN (ICCV'2017)</a></li>
        </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
<tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Inpainting</b>
      </td>
      <td>
        <b>Matting</b>
      </td>
      <td>
        <b>Text-to-Image(Video)</b>
      </td>
      <td>
        <b>3D-aware Generation</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
          <li><a href="configs/global_local/README.md">Global&Local (ToG'2017)</a></li>
          <li><a href="configs/deepfillv1/README.md">DeepFillv1 (CVPR'2018)</a></li>
          <li><a href="configs/partial_conv/README.md">PConv (ECCV'2018)</a></li>
          <li><a href="configs/deepfillv2/README.md">DeepFillv2 (CVPR'2019)</a></li>
          <li><a href="configs/aot_gan/README.md">AOT-GAN (TVCG'2019)</a></li>
          <li><a href="configs/stable_diffusion/README.md">Stable Diffusion Inpainting (CVPR'2022)</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/dim/README.md">DIM (CVPR'2017)</a></li>
          <li><a href="configs/indexnet/README.md">IndexNet (ICCV'2019)</a></li>
          <li><a href="configs/gca/README.md">GCA (AAAI'2020)</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="projects/glide/configs/README.md">GLIDE (NeurIPS'2021)</a></li>
          <li><a href="configs/guided_diffusion/README.md">Guided Diffusion (NeurIPS'2021)</a></li>
          <li><a href="configs/disco_diffusion/README.md">Disco-Diffusion (2022)</a></li>
          <li><a href="configs/stable_diffusion/README.md">Stable-Diffusion (2022)</a></li>
          <li><a href="configs/dreambooth/README.md">DreamBooth (2022)</a></li>
          <li><a href="configs/textual_inversion/README.md">Textual Inversion (2022)</a></li>
          <li><a href="projects/prompt_to_prompt/README.md">Prompt-to-Prompt (2022)</a></li>
          <li><a href="projects/prompt_to_prompt/README.md">Null-text Inversion (2022)</a></li>
          <li><a href="configs/controlnet/README.md">ControlNet (2023)</a></li>
          <li><a href="configs/controlnet_animation/README.md">ControlNet Animation (2023)</a></li>
          <li><a href="configs/stable_diffusion_xl/README.md">Stable Diffusion XL (2023)</a></li>
          <li><a href="configs/animatediff/README.md">AnimateDiff (2023)</a></li>
          <li><a href="configs/vico/README.md">ViCo (2023)</a></li>
          <li><a href="configs/fastcomposer/README.md">FastComposer (2023)</a></li>
          <li><a href="projects/powerpaint/README.md">PowerPaint (2023)</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/eg3d/README.md">EG3D (CVPR'2022)</a></li>
        </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

Please refer to [model_zoo](https://mmagic.readthedocs.io/en/latest/model_zoo/overview.html) for more details.

<p align="right"><a href="#table">🔝Back to Table of Contents</a></p>

## 🤝 Acknowledgement

MMagic is an open source project that is contributed by researchers and engineers from various colleges and companies. We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new methods.

We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks. Thank you all!

<a href="https://github.com/open-mmlab/mmagic/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=open-mmlab/mmagic" />
</a>

<p align="right"><a href="#table">🔝Back to Table of Contents</a></p>

## 🖊️ Citation

If MMagic is helpful to your research, please cite it as below.

```bibtex
@misc{mmagic2023,
    title = {{MMagic}: {OpenMMLab} Multimodal Advanced, Generative, and Intelligent Creation Toolbox},
    author = {{MMagic Contributors}},
    howpublished = {\url{https://github.com/open-mmlab/mmagic}},
    year = {2023}
}
```

```bibtex
@misc{mmediting2022,
    title = {{MMEditing}: {OpenMMLab} Image and Video Editing Toolbox},
    author = {{MMEditing Contributors}},
    howpublished = {\url{https://github.com/open-mmlab/mmediting}},
    year = {2022}
}
```

<p align="right"><a href="#table">🔝Back to Table of Contents</a></p>

## 🎫 License

This project is released under the [Apache 2.0 license](LICENSE).
Please refer to [LICENSES](LICENSE) for the careful check, if you are using our code for commercial matters.

<p align="right"><a href="#table">🔝Back to Table of Contents</a></p>

## 🏗️ ️OpenMMLab Family

- [MMEngine](https://github.com/open-mmlab/mmengine): OpenMMLab foundational library for training deep learning models.
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM installs OpenMMLab packages.
- [MMPreTrain](https://github.com/open-mmlab/mmpretrain): OpenMMLab Pre-training Toolbox and Benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab rotated object detection toolbox and benchmark.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab text detection, recognition, and understanding toolbox.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D human parametric model toolbox and benchmark.
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab self-supervised learning toolbox and benchmark.
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab model compression toolbox and benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab fewshot learning toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
- [MMagic](https://github.com/open-mmlab/mmagic): OpenMMLab Multimodal Advanced, Generative, and Intelligent Creation Toolbox.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab model deployment framework.

<p align="right"><a href="#table">🔝Back to Table of Contents</a></p>
