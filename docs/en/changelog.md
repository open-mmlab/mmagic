# Changelog

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
