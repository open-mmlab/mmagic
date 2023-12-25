# Changelog

**Highlights**

- An advanced and powerful inpainting algorithm named PowerPaint is released in our repository. [Click to View](https://github.com/open-mmlab/mmagic/tree/main/projects/powerpaint)

<div align=center>
<img src="https://github.com/open-mmlab/mmagic/assets/12782558/eba2c6a4-3ff4-4075-a027-0e9799769bf9"/>
</div>

**New Features & Improvements**

- \[Release\] Post release for v1.1.0 by @liuwenran in https://github.com/open-mmlab/mmagic/pull/2043
- \[CodeCamp2023-645\]Add dreambooth new cfg by @YanxingLiu in https://github.com/open-mmlab/mmagic/pull/2042
- \[Enhance\] add new config for _base_ dir by @liuwenran in https://github.com/open-mmlab/mmagic/pull/2053
- \[Enhance\] support using from_pretrained for instance_crop by @zengyh1900 in https://github.com/open-mmlab/mmagic/pull/2066
- \[Enhance\] update support for latest diffusers with lora by @zengyh1900 in https://github.com/open-mmlab/mmagic/pull/2067
- \[Feature\] PowerPaint by @zhuang2002 in https://github.com/open-mmlab/mmagic/pull/2076
- \[Enhance\] powerpaint improvement by @liuwenran in https://github.com/open-mmlab/mmagic/pull/2078
- \[Enhance\] Improve powerpaint by @liuwenran in https://github.com/open-mmlab/mmagic/pull/2080
- \[Enhance\] add outpainting to gradio_PowerPaint.py by @zhuang2002 in https://github.com/open-mmlab/mmagic/pull/2084
- \[MMSIG\] Add new configuration files for StyleGAN2 by @xiaomile in https://github.com/open-mmlab/mmagic/pull/2057
- \[MMSIG\] \[Doc\] Update data_preprocessor.md by @jinxianwei in https://github.com/open-mmlab/mmagic/pull/2055
- \[Enhance\] Enhance PowerPaint by @zhuang2002 in https://github.com/open-mmlab/mmagic/pull/2093

**Bug Fixes**

- \[Fix\] Update README.md by @eze1376 in https://github.com/open-mmlab/mmagic/pull/2048
- \[Fix\] Fix test tokenizer by @liuwenran in https://github.com/open-mmlab/mmagic/pull/2050
- \[Fix\] fix readthedocs building by @liuwenran in https://github.com/open-mmlab/mmagic/pull/2052
- \[Fix\] --local-rank for PyTorch >= 2.0.0 by @youqingxiaozhua in https://github.com/open-mmlab/mmagic/pull/2051
- \[Fix\] animatediff download from openxlab by @JianxinDong in https://github.com/open-mmlab/mmagic/pull/2061
- \[Fix\] fix best practice by @liuwenran in https://github.com/open-mmlab/mmagic/pull/2063
- \[Fix\] try import expand mask from transformers by @zengyh1900 in https://github.com/open-mmlab/mmagic/pull/2064
- \[Fix\] Update diffusers to v0.23.0 by @liuwenran in https://github.com/open-mmlab/mmagic/pull/2069
- \[Fix\] add openxlab link to powerpaint by @liuwenran in https://github.com/open-mmlab/mmagic/pull/2082
- \[Fix\] Update swinir_x2s48w8d6e180_8xb4-lr2e-4-500k_div2k.py, use MultiValLoop. by @ashutoshsingh0223 in https://github.com/open-mmlab/mmagic/pull/2085
- \[Fix\] Fix a test expression that has a logical short circuit. by @munahaf in https://github.com/open-mmlab/mmagic/pull/2046
- \[Fix\] Powerpaint to load safetensors by @sdbds in https://github.com/open-mmlab/mmagic/pull/2088

**New Contributors**

- @eze1376 made their first contribution in https://github.com/open-mmlab/mmagic/pull/2048
- @youqingxiaozhua made their first contribution in https://github.com/open-mmlab/mmagic/pull/2051
- @JianxinDong made their first contribution in https://github.com/open-mmlab/mmagic/pull/2061
- @zhuang2002 made their first contribution in https://github.com/open-mmlab/mmagic/pull/2076
- @ashutoshsingh0223 made their first contribution in https://github.com/open-mmlab/mmagic/pull/2085
- @jinxianwei made their first contribution in https://github.com/open-mmlab/mmagic/pull/2055
- @munahaf made their first contribution in https://github.com/open-mmlab/mmagic/pull/2046
- @sdbds made their first contribution in https://github.com/open-mmlab/mmagic/pull/2088

**Full Changelog**: https://github.com/open-mmlab/mmagic/compare/v1.1.0...v1.2.0

## v1.1.0 (22/09/2023)

**Highlights**

In this new version of MMagic, we have added support for the following five new algorithms.

- Support ViCo, a new SD personalization method. [Click to View](https://github.com/open-mmlab/mmagic/blob/main/configs/vico/README.md)

<table align="center">
<thead>
  <tr>
    <td>
<div align="center">
  <img src="https://github.com/open-mmlab/mmagic/assets/71176040/58a6953c-053a-40ea-8826-eee428c992b5" width="800"/>
  <br/>
</thead>
</table>

- Support AnimateDiff, a popular text2animation method. [Click to View](https://github.com/open-mmlab/mmagic/blob/main/configs/animatediff/README.md)

![512](https://github.com/ElliotQi/mmagic/assets/46469021/54d92aca-dfa9-4eeb-ba38-3f6c981e5399)

- Support SDXL. [Click to View](https://github.com/open-mmlab/mmagic/blob/main/configs/stable_diffusion_xl/README.md)

<div align=center>
<img src="https://github.com/okotaku/diffengine/assets/24734142/27d4ebad-5705-4500-826f-41f425a08c0d"/>
</div>

- Support DragGAN implementation with MMagic. [Click to View](https://github.com/open-mmlab/mmagic/blob/main/configs/draggan/README.md)

<div align=center>
<img src="https://github.com/open-mmlab/mmagic/assets/55343765/7c397bd0-fa07-48fe-8a7c-a4022907404b"/>
</div>

- Support for FastComposer. [Click to View](https://github.com/open-mmlab/mmagic/blob/main/configs/fastcomposer/README.md)

<div align=center>
<img src="https://user-images.githubusercontent.com/14927720/265914135-8a25789c-8d30-40cb-8ac5-e3bd3b617aac.png">
</div>

**New Features & Improvements**

- \[Feature\] Support inference with diffusers pipeline, sd_xl first. by @liuwenran in https://github.com/open-mmlab/mmagic/pull/2023
- \[Enhance\] add negative prompt for sd inferencer by @liuwenran in https://github.com/open-mmlab/mmagic/pull/2021
- \[Enhance\] Update flake8 checking config in setup.cfg by @LeoXing1996 in https://github.com/open-mmlab/mmagic/pull/2007
- \[Enhance\] Add ‘config_name' as a supplement to the 'model_setting' by @liuwenran in https://github.com/open-mmlab/mmagic/pull/2027
- \[Enhance\] faster test by @okotaku in https://github.com/open-mmlab/mmagic/pull/2034
- \[Enhance\] Add OpenXLab Badge by @ZhaoQiiii in https://github.com/open-mmlab/mmagic/pull/2037

**CodeCamp Contributions**

- \[CodeCamp2023-643\] Add new configs of BigGAN by @limafang in https://github.com/open-mmlab/mmagic/pull/2003
- \[CodeCamp2023-648\] MMagic new config GuidedDiffusion by @ooooo-create in https://github.com/open-mmlab/mmagic/pull/2005
- \[CodeCamp2023-649\] MMagic new config Instance Colorization by @ooooo-create in https://github.com/open-mmlab/mmagic/pull/2010
- \[CodeCamp2023-652\] MMagic new config StyleGAN3 by @hhy150 in https://github.com/open-mmlab/mmagic/pull/2018
- \[CodeCamp2023-653\] Add new configs of Real BasicVSR by @RangeKing in https://github.com/open-mmlab/mmagic/pull/2030

**Bug Fixes**

- \[Fix\] Fix best practice and back to contents on mainpage, add new models to model zoo by @liuwenran in https://github.com/open-mmlab/mmagic/pull/2001
- \[Fix\] Check CI error and remove main stream gpu test by @liuwenran in https://github.com/open-mmlab/mmagic/pull/2013
- \[Fix\] Check circle ci memory by @liuwenran in https://github.com/open-mmlab/mmagic/pull/2016
- \[Fix\] remove code and fix clip loss ut test by @liuwenran in https://github.com/open-mmlab/mmagic/pull/2017
- \[Fix\] mock infer in diffusers pipeline inferencer ut. by @liuwenran in https://github.com/open-mmlab/mmagic/pull/2026
- \[Fix\] Fix bug caused by merging draggan by @liuwenran in https://github.com/open-mmlab/mmagic/pull/2029
- \[Fix\] Update QRcode by @crazysteeaam in https://github.com/open-mmlab/mmagic/pull/2009
- \[Fix\] Replace the download links in README with OpenXLab version by @FerryHuang in https://github.com/open-mmlab/mmagic/pull/2038
- \[Fix\] Increase docstring coverage by @liuwenran in https://github.com/open-mmlab/mmagic/pull/2039

**New Contributors**

- @limafang made their first contribution in https://github.com/open-mmlab/mmagic/pull/2003
- @ooooo-create made their first contribution in https://github.com/open-mmlab/mmagic/pull/2005
- @hhy150 made their first contribution in https://github.com/open-mmlab/mmagic/pull/2018
- @ZhaoQiiii made their first contribution in https://github.com/open-mmlab/mmagic/pull/2037
- @ElliotQi made their first contribution in https://github.com/open-mmlab/mmagic/pull/1980
- @Beaconsyh08 made their first contribution in https://github.com/open-mmlab/mmagic/pull/2012

**Full Changelog**: https://github.com/open-mmlab/mmagic/compare/v1.0.2...v1.0.3

## v1.0.2 (24/08/2023)

**Highlights**

**1. More detailed documentation**

Thank you to the community contributors for helping us improve the documentation. We have improved many documents, including both Chinese and English versions. Please refer to the [documentation](https://mmagic.readthedocs.io/en/latest/) for more details.

**2. New algorithms**

- Support Prompt-to-prompt, DDIM Inversion and Null-text Inversion. [Click to View.](https://github.com/open-mmlab/mmagic/blob/main/projects/prompt_to_prompt/README.md)

From right to left: origin image, DDIM inversion, Null-text inversion

<center class="half">
    <img src="https://github.com/FerryHuang/mmagic/assets/71176040/34d8a467-5378-41fb-83c6-b23c9dee8f0a" width="200"/><img src="https://github.com/FerryHuang/mmagic/assets/71176040/3d3814b4-7fb5-4232-a56f-fd7fef0ba28e" width="200"/><img src="https://github.com/FerryHuang/mmagic/assets/71176040/43008ed4-a5a3-4f81-ba9f-95d9e79e6a08" width="200"/>
</center>

Prompt-to-prompt Editing

<div align="center">
  <b>cat -> dog</b>
  <br/>
  <img src="https://github.com/FerryHuang/mmagic/assets/71176040/f5d3fc0c-aa7b-4525-9364-365b254d51ca" width="500"/>
</div>

<div align="center">
  <b>spider man -> iron man(attention replace)</b>
  <br/>
  <img src="https://github.com/FerryHuang/mmagic/assets/71176040/074adbc6-bd48-4c82-99aa-f322cf937f5a" width="500"/>
</div>

<div align="center">
  <b>Effel tower -> Effel tower at night (attention refine)</b>
  <br/>
  <img src="https://github.com/FerryHuang/mmagic/assets/71176040/f815dab3-b20c-4936-90e3-a060d3717e22" width="500"/>
</div>

<div align="center">
  <b>blossom sakura tree -> blossom(-3) sakura tree (attention reweight)</b>
  <br/>
  <img src="https://github.com/FerryHuang/mmagic/assets/71176040/5ef770b9-4f28-4ae7-84b0-6c15ea7450e9" width="500"/>
</div>

- Support Textual Inversion. [Click to view.](https://github.com/open-mmlab/mmagic/blob/main/configs/textual_inversion/README.md)

<div align=center>
<img src="https://github.com/open-mmlab/mmagic/assets/28132635/b2dac6f1-5151-4199-bcc2-71b5b1523a16">
</div>

- Support Attention Injection for more stable video generation with controlnet. [Click to view.](https://github.com/open-mmlab/mmagic/blob/main/configs/controlnet_animation/README.md)
- Support Stable Diffusion Inpainting. [Click to view.](https://github.com/open-mmlab/mmagic/blob/main/configs/stable_diffusion/README.md)

**New Features & Improvements**

- \[Enhancement\] Support noise offset in stable diffusion training by @LeoXing1996 in https://github.com/open-mmlab/mmagic/pull/1880
- \[Community\] Support Glide Upsampler by @Taited in https://github.com/open-mmlab/mmagic/pull/1663
- \[Enhance\] support controlnet inferencer by @Z-Fran in https://github.com/open-mmlab/mmagic/pull/1891
- \[Feature\] support Albumentations augmentation transformations and pipeline by @Z-Fran in https://github.com/open-mmlab/mmagic/pull/1894
- \[Feature\] Add Attention Injection for unet by @liuwenran in https://github.com/open-mmlab/mmagic/pull/1895
- \[Enhance\] update benchmark scripts by @Z-Fran in https://github.com/open-mmlab/mmagic/pull/1907
- \[Enhancement\] update mmagic docs by @crazysteeaam in https://github.com/open-mmlab/mmagic/pull/1920
- \[Enhancement\] Support Prompt-to-prompt, ddim inversion and null-text inversion by @FerryHuang in https://github.com/open-mmlab/mmagic/pull/1908
- \[CodeCamp2023-302\] Support MMagic visualization and write a user guide  by @aptsunny in https://github.com/open-mmlab/mmagic/pull/1939
- \[Feature\] Support Textual Inversion by @LeoXing1996 in https://github.com/open-mmlab/mmagic/pull/1822
- \[Feature\] Support stable diffusion inpaint by @Taited in https://github.com/open-mmlab/mmagic/pull/1976
- \[Enhancement\] Adopt `BaseModule` for some models by @LeoXing1996 in https://github.com/open-mmlab/mmagic/pull/1543
- \[MMSIG\]支持 DeblurGANv2 inference by @xiaomile in https://github.com/open-mmlab/mmagic/pull/1955
- \[CodeCamp2023-647\] Add new configs of EG3D by @RangeKing in https://github.com/open-mmlab/mmagic/pull/1985

**Bug Fixes**

- Fix dtype error in StableDiffusion and DreamBooth training by @LeoXing1996 in https://github.com/open-mmlab/mmagic/pull/1879
- Fix gui VideoSlider bug by @Z-Fran in https://github.com/open-mmlab/mmagic/pull/1885
- Fix init_model and glide demo by @Z-Fran in https://github.com/open-mmlab/mmagic/pull/1888
- Fix InstColorization bug when dim=3 by @Z-Fran in https://github.com/open-mmlab/mmagic/pull/1901
- Fix sd and controlnet fp16 bugs by @Z-Fran in https://github.com/open-mmlab/mmagic/pull/1914
- Fix num_images_per_prompt in controlnet by @LeoXing1996 in https://github.com/open-mmlab/mmagic/pull/1936
- Revise metafile for sd-inpainting to fix inferencer init by @LeoXing1996 in https://github.com/open-mmlab/mmagic/pull/1995

**New Contributors**

- @wyyang23 made their first contribution in https://github.com/open-mmlab/mmagic/pull/1886
- @yehuixie made their first contribution in https://github.com/open-mmlab/mmagic/pull/1912
- @crazysteeaam made their first contribution in https://github.com/open-mmlab/mmagic/pull/1920
- @BUPT-NingXinyu made their first contribution in https://github.com/open-mmlab/mmagic/pull/1921
- @zhjunqin made their first contribution in https://github.com/open-mmlab/mmagic/pull/1918
- @xuesheng1031 made their first contribution in https://github.com/open-mmlab/mmagic/pull/1923
- @wslgqq277g made their first contribution in https://github.com/open-mmlab/mmagic/pull/1934
- @LYMDLUT made their first contribution in https://github.com/open-mmlab/mmagic/pull/1933
- @RangeKing made their first contribution in https://github.com/open-mmlab/mmagic/pull/1930
- @xin-li-67 made their first contribution in https://github.com/open-mmlab/mmagic/pull/1932
- @chg0901 made their first contribution in https://github.com/open-mmlab/mmagic/pull/1931
- @aptsunny made their first contribution in https://github.com/open-mmlab/mmagic/pull/1939
- @YanxingLiu made their first contribution in https://github.com/open-mmlab/mmagic/pull/1943
- @tackhwa made their first contribution in https://github.com/open-mmlab/mmagic/pull/1937
- @Geo-Chou made their first contribution in https://github.com/open-mmlab/mmagic/pull/1940
- @qsun1 made their first contribution in https://github.com/open-mmlab/mmagic/pull/1956
- @ththth888 made their first contribution in https://github.com/open-mmlab/mmagic/pull/1961
- @sijiua made their first contribution in https://github.com/open-mmlab/mmagic/pull/1967
- @MING-ZCH made their first contribution in https://github.com/open-mmlab/mmagic/pull/1982
- @AllYoung made their first contribution in https://github.com/open-mmlab/mmagic/pull/1996

## v1.0.1 (26/05/2023)

**New Features & Improvements**

- Support tomesd for StableDiffusion speed-up. [#1801](https://github.com/open-mmlab/mmagic/pull/1801)
- Support all inpainting/matting/image restoration models inferencer. [#1833](https://github.com/open-mmlab/mmagic/pull/1833), [#1873](https://github.com/open-mmlab/mmagic/pull/1873)
- Support animated drawings at projects. [#1837](https://github.com/open-mmlab/mmagic/pull/1837)
- Support Style-Based Global Appearance Flow for Virtual Try-On at projects. [#1786](https://github.com/open-mmlab/mmagic/pull/1786)
- Support tokenizer wrapper and support EmbeddingLayerWithFixe. [#1846](https://github.com/open-mmlab/mmagic/pull/1846)

**Bug Fixes**

- Fix install requirements. [#1819](https://github.com/open-mmlab/mmagic/pull/1819)
- Fix inst-colorization PackInputs. [#1828](https://github.com/open-mmlab/mmagic/pull/1828), [#1827](https://github.com/open-mmlab/mmagic/pull/1827)
- Fix inferencer in pip-install. [#1875](https://github.com/open-mmlab/mmagic/pull/1875)

**New Contributors**

- @XDUWQ made their first contribution in https://github.com/open-mmlab/mmagic/pull/1830
- @FerryHuang made their first contribution in https://github.com/open-mmlab/mmagic/pull/1786
- @bobo0810 made their first contribution in https://github.com/open-mmlab/mmagic/pull/1851
- @jercylew made their first contribution in https://github.com/open-mmlab/mmagic/pull/1874

## v1.0.0 (25/04/2023)

We are excited to announce the release of MMagic v1.0.0 that inherits from [MMEditing](https://github.com/open-mmlab/mmediting) and [MMGeneration](https://github.com/open-mmlab/mmgeneration).

![mmagic-log](https://user-images.githubusercontent.com/49083766/233557648-9034f5a0-c85d-4092-b700-3a28072251b6.png)

Since its inception, MMEditing has been the preferred algorithm library for many super-resolution, editing, and generation tasks, helping research teams win more than 10 top international competitions and supporting over 100 GitHub ecosystem projects. After iterative updates with OpenMMLab 2.0 framework and merged with MMGeneration, MMEditing has become a powerful tool that supports low-level algorithms based on both GAN and CNN.

Today, MMEditing embraces Generative AI and transforms into a more advanced and comprehensive AIGC toolkit: **MMagic** (**M**ultimodal **A**dvanced, **G**enerative, and **I**ntelligent **C**reation).

In MMagic, we have supported 53+ models in multiple tasks such as fine-tuning for stable diffusion, text-to-image, image and video restoration, super-resolution, editing and generation. With excellent training and experiment management support from [MMEngine](https://github.com/open-mmlab/mmengine), MMagic will provide more agile and flexible experimental support for researchers and AIGC enthusiasts, and help you on your AIGC exploration journey. With MMagic, experience more magic in generation! Let's open a new era beyond editing together. More than Editing, Unlock the Magic!

**Highlights**

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

https://user-images.githubusercontent.com/49083766/233564593-7d3d48ed-e843-4432-b610-35e3d257765c.mp4

**2. Magic Diffusion Model**

For the Diffusion Model, we provide the following "magic" :

- Support image generation based on Stable Diffusion and Disco Diffusion.

- Support Finetune methods such as Dreambooth and DreamBooth LoRA.

- Support controllability in text-to-image generation using ControlNet.
  ![de87f16f-bf6d-4a61-8406-5ecdbb9167b6](https://user-images.githubusercontent.com/49083766/233558077-2005e603-c5a8-49af-930f-e7a465ca818b.png)

- Support acceleration and optimization strategies based on xFormers to improve training and inference efficiency.

- Support video generation based on MultiFrame Render.
  MMagic supports the generation of long videos in various styles through ControlNet and MultiFrame Render.
  prompt keywords: a handsome man, silver hair, smiling, play basketball

  https://user-images.githubusercontent.com/12782558/227149757-fd054d32-554f-45d5-9f09-319184866d85.mp4

  prompt keywords: a girl, black hair, white pants, smiling, play basketball

  https://user-images.githubusercontent.com/49083766/233559964-bd5127bd-52f6-44b6-a089-9d7adfbc2430.mp4

  prompt keywords: a handsome man

  https://user-images.githubusercontent.com/12782558/227152129-d70d5f76-a6fc-4d23-97d1-a94abd08f95a.mp4

- Support calling basic models and sampling strategies through DiffuserWrapper.

- SAM + MMagic = Generate Anything！
  SAM (Segment Anything Model) is a popular model these days and can also provide more support for MMagic! If you want to create your own animation, you can go to [OpenMMLab PlayGround](https://github.com/open-mmlab/playground/blob/main/mmediting_sam/README.md).

  https://user-images.githubusercontent.com/49083766/233562228-f39fc675-326c-4ae8-986a-c942059effd0.mp4

**3. Upgraded Framework**

To improve your "spellcasting" efficiency, we have made the following adjustments to the "magic circuit":

- By using MMEngine and MMCV of OpenMMLab 2.0 framework, We decompose the editing framework into different modules and one can easily construct a customized editor framework by combining different modules. We can define the training process just like playing with Legos and provide rich components and strategies. In MMagic, you can complete controls on the training process with different levels of APIs.
- Support for 33+ algorithms accelerated by Pytorch 2.0.
- Refactor DataSample to support the combination and splitting of batch dimensions.
- Refactor DataPreprocessor and unify the data format for various tasks during training and inference.
- Refactor MultiValLoop and MultiTestLoop, supporting the evaluation of both generation-type metrics (e.g. FID) and reconstruction-type metrics (e.g. SSIM), and supporting the evaluation of multiple datasets at once.
- Support visualization on local files or using tensorboard and wandb.

**New Features & Improvements**

- Support 53+ algorithms, 232+ configs, 213+ checkpoints, 26+ loss functions, and 20+ metrics.
- Support controlnet animation and Gradio gui. [Click to view.](https://github.com/open-mmlab/mmagic/tree/main/configs/controlnet_animation)
- Support Inferencer and Demo using High-level Inference APIs. [Click to view.](https://github.com/open-mmlab/mmagic/tree/main/demo)
- Support Gradio gui of Inpainting inference. [Click to view.](https://github.com/open-mmlab/mmagic/blob/main/demo/gradio-demo.py)
- Support qualitative comparison tools. [Click to view.](https://github.com/open-mmlab/mmagic/tree/main/tools/gui)
- Enable projects. [Click to view.](https://github.com/open-mmlab/mmagic/tree/main/projects)
- Improve converters scripts and documents for datasets. [Click to view.](https://github.com/open-mmlab/mmagic/tree/main/tools/dataset_converters)

## v1.0.0rc7 (07/04/2023)

**Highlights**

We are excited to announce the release of MMEditing 1.0.0rc7. This release supports 51+ models, 226+ configs and 212+ checkpoints in MMGeneration and MMEditing. We highlight the following new features

- Support DiffuserWrapper
- Support ControlNet (training and inference).
- Support PyTorch 2.0.

**New Features & Improvements**

- Support DiffuserWrapper. [#1692](https://github.com/open-mmlab/mmagic/pull/1692)
- Support ControlNet (training and inference). [#1744](https://github.com/open-mmlab/mmagic/pull/1744)
- Support PyTorch 2.0 (successfully compile 33+ models on 'inductor' backend). [#1742](https://github.com/open-mmlab/mmagic/pull/1742)
- Support Image Super-Resolution and Video Super-Resolution models inferencer. [#1662](https://github.com/open-mmlab/mmagic/pull/1662), [#1720](https://github.com/open-mmlab/mmagic/pull/1720)
- Refactor tools/get_flops script. [#1675](https://github.com/open-mmlab/mmagic/pull/1675)
- Refactor dataset_converters and documents for datasets. [#1690](https://github.com/open-mmlab/mmagic/pull/1690)
- Move stylegan ops to MMCV. [#1383](https://github.com/open-mmlab/mmagic/pull/1383)

**Bug Fixes**

- Fix disco inferencer. [#1673](https://github.com/open-mmlab/mmagic/pull/1673)
- Fix nafnet optimizer config. [#1716](https://github.com/open-mmlab/mmagic/pull/1716)
- Fix tof typo. [#1711](https://github.com/open-mmlab/mmagic/pull/1711)

**Contributors**

A total of 8 developers contributed to this release.
Thanks @LeoXing1996, @Z-Fran, @plyfager, @zengyh1900, @liuwenran, @ryanxingql, @HAOCHENYE, @VongolaWu

**New Contributors**

- @HAOCHENYE made their first contribution in https://github.com/open-mmlab/mmagic/pull/1712

## v1.0.0rc6 (02/03/2023)

**Highlights**

We are excited to announce the release of MMEditing 1.0.0rc6. This release supports 50+ models, 222+ configs and 209+ checkpoints in MMGeneration and MMEditing. We highlight the following new features

- Support Gradio gui of Inpainting inference.
- Support Colorization, Translationin and GAN models inferencer.

**New Features & Improvements**

- Refactor FileIO. [#1572](https://github.com/open-mmlab/mmagic/pull/1572)
- Refactor registry. [#1621](https://github.com/open-mmlab/mmagic/pull/1621)
- Refactor Random degradations. [#1583](https://github.com/open-mmlab/mmagic/pull/1583)
- Refactor DataSample, DataPreprocessor, Metric and Loop. [#1656](https://github.com/open-mmlab/mmagic/pull/1656)
- Use mmengine.basemodule instead of nn.module. [#1491](https://github.com/open-mmlab/mmagic/pull/1491)
- Refactor Main Page. [#1609](https://github.com/open-mmlab/mmagic/pull/1609)
- Support Gradio gui of Inpainting inference. [#1601](https://github.com/open-mmlab/mmagic/pull/1601)
- Support Colorization inferencer. [#1588](https://github.com/open-mmlab/mmagic/pull/1588)
- Support Translation models inferencer. [#1650](https://github.com/open-mmlab/mmagic/pull/1650)
- Support GAN models inferencer. [#1653](https://github.com/open-mmlab/mmagic/pull/1653), [#1659](https://github.com/open-mmlab/mmagic/pull/1659)
- Print config tool. [#1590](https://github.com/open-mmlab/mmagic/pull/1590)
- Improve type hints. [#1604](https://github.com/open-mmlab/mmagic/pull/1604)
- Update Chinese documents of metrics and datasets. [#1568](https://github.com/open-mmlab/mmagic/pull/1568), [#1638](https://github.com/open-mmlab/mmagic/pull/1638)
- Update Chinese documents of BigGAN and Disco-Diffusion. [#1620](https://github.com/open-mmlab/mmagic/pull/1620)
- Update Evaluation and README of Guided-Diffusion. [#1547](https://github.com/open-mmlab/mmagic/pull/1547)

**Bug Fixes**

- Fix the meaning of `momentum` in EMA. [#1581](https://github.com/open-mmlab/mmagic/pull/1581)
- Fix output dtype of RandomNoise. [#1585](https://github.com/open-mmlab/mmagic/pull/1585)
- Fix pytorch2onnx tool. [#1629](https://github.com/open-mmlab/mmagic/pull/1629)
- Fix API documents. [#1641](https://github.com/open-mmlab/mmagic/pull/1641), [#1642](https://github.com/open-mmlab/mmagic/pull/1642)
- Fix loading RealESRGAN EMA weights. [#1647](https://github.com/open-mmlab/mmagic/pull/1647)
- Fix arg passing bug of dataset_converters scripts. [#1648](https://github.com/open-mmlab/mmagic/pull/1648)

**Contributors**

A total of 17 developers contributed to this release.
Thanks @plyfager, @LeoXing1996, @Z-Fran, @zengyh1900, @VongolaWu, @liuwenran, @austinmw, @dienachtderwelt, @liangzelong, @i-aki-y, @xiaomile, @Li-Qingyun, @vansin, @Luo-Yihang, @ydengbi, @ruoningYu, @triple-Mu

**New Contributors**

- @dienachtderwelt made their first contribution in https://github.com/open-mmlab/mmagic/pull/1578
- @i-aki-y made their first contribution in https://github.com/open-mmlab/mmagic/pull/1590
- @triple-Mu made their first contribution in https://github.com/open-mmlab/mmagic/pull/1618
- @Li-Qingyun made their first contribution in https://github.com/open-mmlab/mmagic/pull/1640
- @Luo-Yihang made their first contribution in https://github.com/open-mmlab/mmagic/pull/1648
- @ydengbi made their first contribution in https://github.com/open-mmlab/mmagic/pull/1557

## v1.0.0rc5 (04/01/2023)

**Highlights**

We are excited to announce the release of MMEditing 1.0.0rc5. This release supports 49+ models, 180+ configs and 177+ checkpoints in MMGeneration and MMEditing. We highlight the following new features

- Support Restormer.
- Support GLIDE.
- Support SwinIR.
- Support Stable Diffusion.

**New Features & Improvements**

- Disco notebook. (#1507)
- Revise test requirements and CI. (#1514)
- Recursive generate summary and docstring. (#1517)
- Enable projects. (#1526)
- Support mscoco dataset. (#1520)
- Improve Chinese documents. (#1532)
- Type hints. (#1481)
- Update download link of checkpoints. (#1554)
- Update deployment guide. (#1551)

**Bug Fixes**

- Fix documentation link checker. (#1522)
- Fix ssim first channel bug. (#1515)
- Fix extract_gt_data of realesrgan. (#1542)
- Fix model index. (#1559)
- Fix config path in disco-diffusion. (#1553)
- Fix text2image inferencer. (#1523)

**Contributors**

A total of 16 developers contributed to this release.
Thanks @plyfager, @LeoXing1996, @Z-Fran, @zengyh1900, @VongolaWu, @liuwenran, @AlexZou14, @lvhan028, @xiaomile, @ldr426, @austin273, @whu-lee, @willaty, @curiosity654, @Zdafeng, @Taited

**New Contributors**

- @xiaomile made their first contribution in https://github.com/open-mmlab/mmagic/pull/1481
- @ldr426 made their first contribution in https://github.com/open-mmlab/mmagic/pull/1542
- @austin273 made their first contribution in https://github.com/open-mmlab/mmagic/pull/1553
- @whu-lee made their first contribution in https://github.com/open-mmlab/mmagic/pull/1539
- @willaty made their first contribution in https://github.com/open-mmlab/mmagic/pull/1541
- @curiosity654 made their first contribution in https://github.com/open-mmlab/mmagic/pull/1556
- @Zdafeng made their first contribution in https://github.com/open-mmlab/mmagic/pull/1476
- @Taited made their first contribution in https://github.com/open-mmlab/mmagic/pull/1534

## v1.0.0rc4 (05/12/2022)

**Highlights**

We are excited to announce the release of MMEditing 1.0.0rc4. This release supports 45+ models, 176+ configs and 175+ checkpoints in MMGeneration and MMEditing. We highlight the following new features

- Support High-level APIs.
- Support diffusion models.
- Support Text2Image Task.
- Support 3D-Aware Generation.

**New Features & Improvements**

- Refactor High-level APIs. (#1410)
- Support disco-diffusion text-2-image. (#1234, #1504)
- Support EG3D. (#1482, #1493, #1494, #1499)
- Support NAFNet model. (#1369)

**Bug Fixes**

- fix srgan train config. (#1441)
- fix cain config. (#1404)
- fix rdn and srcnn train configs. (#1392)

**Contributors**

A total of 14 developers contributed to this release.
Thanks @plyfager, @LeoXing1996, @Z-Fran, @zengyh1900, @VongolaWu, @gaoyang07, @ChangjianZhao, @zxczrx123, @jackghosts, @liuwenran, @CCODING04, @RoseZhao929, @shaocongliu, @liangzelong.

**New Contributors**

- @gaoyang07 made their first contribution in https://github.com/open-mmlab/mmagic/pull/1372
- @ChangjianZhao made their first contribution in https://github.com/open-mmlab/mmagic/pull/1461
- @zxczrx123 made their first contribution in https://github.com/open-mmlab/mmagic/pull/1462
- @jackghosts made their first contribution in https://github.com/open-mmlab/mmagic/pull/1463
- @liuwenran made their first contribution in https://github.com/open-mmlab/mmagic/pull/1410
- @CCODING04 made their first contribution in https://github.com/open-mmlab/mmagic/pull/783
- @RoseZhao929 made their first contribution in https://github.com/open-mmlab/mmagic/pull/1474
- @shaocongliu made their first contribution in https://github.com/open-mmlab/mmagic/pull/1470
- @liangzelong made their first contribution in https://github.com/open-mmlab/mmagic/pull/1488

## v1.0.0rc3 (10/11/2022)

**Highlights**

We are excited to announce the release of MMEditing 1.0.0rc3. This release supports 43+ models, 170+ configs and 169+ checkpoints in MMGeneration and MMEditing. We highlight the following new features

- convert `mmdet` and `clip` to optional requirements.

**New Features & Improvements**

- Support `try_import` for `mmdet`. (#1408)
- Support `try_import` for `flip`. (#1420)
- Update `.gitignore`. ($1416)
- Set `real_feat` to cpu in `inception_utils`. (#1415)
- Modify README and configs of StyleGAN2 and PEGAN (#1418)
- Improve the rendering of Docs-API (#1373)

**Bug Fixes**

- Revise config and pretrain model loading in ESRGAN (#1407)
- Revise config of LSGAN (#1409)
- Revise config of CAIN (#1404)

**Contributors**

A total of 5 developers contributed to this release.
@Z-Fran, @zengyh1900, @plyfager, @LeoXing1996, @ruoningYu.

## v1.0.0rc2 (02/11/2022)

**Highlights**

We are excited to announce the release of MMEditing 1.0.0rc2. This release supports 43+ models, 170+ configs and 169+ checkpoints in MMGeneration and MMEditing. We highlight the following new features

- patch-based and slider-based image and video comparison viewer.
- image colorization.

**New Features & Improvements**

- Support qualitative comparison tools. (#1303)
- Support instance aware colorization. (#1370)
- Support multi-metrics with different sample-model. (#1171)
- Improve the implementation
  - refactoring evaluation metrics. (#1164)
  - Save gt images in PGGAN's `forward`. (#1332)
  - Improve type and change default number of `preprocess_div2k_dataset.py`. (#1380)
  - Support pixel value clip in visualizer. (#1365)
  - Support SinGAN Dataset and SinGAN demo. (#1363)
  - Avoid cast int and float in GenDataPreprocessor. (#1385)
- Improve the documentation
  - Update a menu switcher. (#1162)
  - Fix TTSR's README. (#1325)

**Bug Fixes**

- Fix PPL bug. (#1172)
- Fix RDN number of channels. (#1328)
- Fix types of exceptions in demos. (#1372)
- Fix realesrgan ema. (#1341)
- Improve the assertion to ensuer `GenerateFacialHeatmap` as `np.float32`. (#1310)
- Fix sampling behavior of `unpaired_dataset.py` and  urls in cyclegan's README. (#1308)
- Fix vsr models in pytorch2onnx. (#1300)
- Fix incorrect settings in configs. (#1167,#1200,#1236,#1293,#1302,#1304,#1319,#1331,#1336,#1349,#1352,#1353,#1358,#1364,#1367,#1384,#1386,#1391,#1392,#1393)

**New Contributors**

- @gaoyang07 made their first contribution in https://github.com/open-mmlab/mmagic/pull/1372

**Contributors**

A total of 7 developers contributed to this release.
Thanks @LeoXing1996, @Z-Fran, @zengyh1900, @plyfager, @ryanxingql, @ruoningYu, @gaoyang07.

## v1.0.0rc1(23/9/2022)

MMEditing 1.0.0rc1 has merged MMGeneration 1.x.

- Support 42+ algorithms, 169+ configs and 168+ checkpoints.
- Support 26+ loss functions, 20+ metrics.
- Support tensorboard, wandb.
- Support unconditional GANs, conditional GANs, image2image translation and internal learning.

## v1.0.0rc0(31/8/2022)

MMEditing 1.0.0rc0 is the first version of MMEditing 1.x, a part of the OpenMMLab 2.0 projects.

Built upon the new [training engine](https://github.com/open-mmlab/mmengine), MMEditing 1.x unifies the interfaces of dataset, models, evaluation, and visualization.

And there are some BC-breaking changes. Please check [the migration tutorial](https://mmagic.readthedocs.io/en/latest/migration/overview.html) for more details.
