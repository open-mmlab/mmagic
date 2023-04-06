# Changelog

## v1.0.0rc7 (06/04/2023)

**Highlights**

We are excited to announce the release of MMEditing 1.0.0rc7. This release supports 50+ models, 222+ configs and 209+ checkpoints in MMGeneration and MMEditing. We highlight the following new features

- Support DiffuserWrapper
- Support ControlNet (training and inference).
- Support PyTorch 2.0.

**New Features & Improvements**

- Support DiffuserWrapper. [#1692](https://github.com/open-mmlab/mmediting/pull/1692)
- Support ControlNet (training and inference). [#1744](https://github.com/open-mmlab/mmediting/pull/1744)
- Support PyTorch 2.0 (successfully compile 33+ models on 'inductor' backend). [#1742](https://github.com/open-mmlab/mmediting/pull/1742)
- Support Image Super-Resolution and Video Super-Resolution models inferencer. [#1662](https://github.com/open-mmlab/mmediting/pull/1662), [#1720](https://github.com/open-mmlab/mmediting/pull/1720)
- Refactor tools/get_flops script. [#1675](https://github.com/open-mmlab/mmediting/pull/1675)
- Refactor dataset_converters and documents for datasets. [#1690](https://github.com/open-mmlab/mmediting/pull/1690)
- Move stylegan ops to MMCV. [#1383](https://github.com/open-mmlab/mmediting/pull/1383)

**Bug Fixes**

- Fix disco inferencer. [#1673](https://github.com/open-mmlab/mmediting/pull/1673)
- Fix nafnet optimizer config. [#1716](https://github.com/open-mmlab/mmediting/pull/1716)
- Fix tof typo. [#1711](https://github.com/open-mmlab/mmediting/pull/1711)

**Contributors**

A total of 8 developers contributed to this release.
Thanks @LeoXing1996, @Z-Fran, @plyfager, @zengyh1900, @liuwenran, @ryanxingql, @HAOCHENYE, @VongolaWu

**New Contributors**

- @HAOCHENYE made their first contribution in https://github.com/open-mmlab/mmediting/pull/1712

## v1.0.0rc6 (02/03/2023)

**Highlights**

We are excited to announce the release of MMEditing 1.0.0rc6. This release supports 50+ models, 222+ configs and 209+ checkpoints in MMGeneration and MMEditing. We highlight the following new features

- Support Gradio gui of Inpainting inference.
- Support Colorization, Translationin and GAN models inferencer.

**New Features & Improvements**

- Refactor FileIO. [#1572](https://github.com/open-mmlab/mmediting/pull/1572)
- Refactor registry. [#1621](https://github.com/open-mmlab/mmediting/pull/1621)
- Refactor Random degradations. [#1583](https://github.com/open-mmlab/mmediting/pull/1583)
- Refactor DataSample, DataPreprocessor, Metric and Loop. [#1656](https://github.com/open-mmlab/mmediting/pull/1656)
- Use mmengine.basemodule instead of nn.module. [#1491](https://github.com/open-mmlab/mmediting/pull/1491)
- Refactor Main Page. [#1609](https://github.com/open-mmlab/mmediting/pull/1609)
- Support Gradio gui of Inpainting inference. [#1601](https://github.com/open-mmlab/mmediting/pull/1601)
- Support Colorization inferencer. [#1588](https://github.com/open-mmlab/mmediting/pull/1588)
- Support Translation models inferencer. [#1650](https://github.com/open-mmlab/mmediting/pull/1650)
- Support GAN models inferencer. [#1653](https://github.com/open-mmlab/mmediting/pull/1653), [#1659](https://github.com/open-mmlab/mmediting/pull/1659)
- Print config tool. [#1590](https://github.com/open-mmlab/mmediting/pull/1590)
- Improve type hints. [#1604](https://github.com/open-mmlab/mmediting/pull/1604)
- Update Chinese documents of metrics and datasets. [#1568](https://github.com/open-mmlab/mmediting/pull/1568), [#1638](https://github.com/open-mmlab/mmediting/pull/1638)
- Update Chinese documents of BigGAN and Disco-Diffusion. [#1620](https://github.com/open-mmlab/mmediting/pull/1620)
- Update Evaluation and README of Guided-Diffusion. [#1547](https://github.com/open-mmlab/mmediting/pull/1547)

**Bug Fixes**

- Fix the meaning of `momentum` in EMA. [#1581](https://github.com/open-mmlab/mmediting/pull/1581)
- Fix output dtype of RandomNoise. [#1585](https://github.com/open-mmlab/mmediting/pull/1585)
- Fix pytorch2onnx tool. [#1629](https://github.com/open-mmlab/mmediting/pull/1629)
- Fix API documents. [#1641](https://github.com/open-mmlab/mmediting/pull/1641), [#1642](https://github.com/open-mmlab/mmediting/pull/1642)
- Fix loading RealESRGAN EMA weights. [#1647](https://github.com/open-mmlab/mmediting/pull/1647)
- Fix arg passing bug of dataset_converters scripts. [#1648](https://github.com/open-mmlab/mmediting/pull/1648)

**Contributors**

A total of 17 developers contributed to this release.
Thanks @plyfager, @LeoXing1996, @Z-Fran, @zengyh1900, @VongolaWu, @liuwenran, @austinmw, @dienachtderwelt, @liangzelong, @i-aki-y, @xiaomile, @Li-Qingyun, @vansin, @Luo-Yihang, @ydengbi, @ruoningYu, @triple-Mu

**New Contributors**

- @dienachtderwelt made their first contribution in https://github.com/open-mmlab/mmediting/pull/1578
- @i-aki-y made their first contribution in https://github.com/open-mmlab/mmediting/pull/1590
- @triple-Mu made their first contribution in https://github.com/open-mmlab/mmediting/pull/1618
- @Li-Qingyun made their first contribution in https://github.com/open-mmlab/mmediting/pull/1640
- @Luo-Yihang made their first contribution in https://github.com/open-mmlab/mmediting/pull/1648
- @ydengbi made their first contribution in https://github.com/open-mmlab/mmediting/pull/1557

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

- @xiaomile made their first contribution in https://github.com/open-mmlab/mmediting/pull/1481
- @ldr426 made their first contribution in https://github.com/open-mmlab/mmediting/pull/1542
- @austin273 made their first contribution in https://github.com/open-mmlab/mmediting/pull/1553
- @whu-lee made their first contribution in https://github.com/open-mmlab/mmediting/pull/1539
- @willaty made their first contribution in https://github.com/open-mmlab/mmediting/pull/1541
- @curiosity654 made their first contribution in https://github.com/open-mmlab/mmediting/pull/1556
- @Zdafeng made their first contribution in https://github.com/open-mmlab/mmediting/pull/1476
- @Taited made their first contribution in https://github.com/open-mmlab/mmediting/pull/1534

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

- @gaoyang07 made their first contribution in https://github.com/open-mmlab/mmediting/pull/1372
- @ChangjianZhao made their first contribution in https://github.com/open-mmlab/mmediting/pull/1461
- @zxczrx123 made their first contribution in https://github.com/open-mmlab/mmediting/pull/1462
- @jackghosts made their first contribution in https://github.com/open-mmlab/mmediting/pull/1463
- @liuwenran made their first contribution in https://github.com/open-mmlab/mmediting/pull/1410
- @CCODING04 made their first contribution in https://github.com/open-mmlab/mmediting/pull/783
- @RoseZhao929 made their first contribution in https://github.com/open-mmlab/mmediting/pull/1474
- @shaocongliu made their first contribution in https://github.com/open-mmlab/mmediting/pull/1470
- @liangzelong made their first contribution in https://github.com/open-mmlab/mmediting/pull/1488

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

- @gaoyang07 made their first contribution in https://github.com/open-mmlab/mmediting/pull/1372

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

And there are some BC-breaking changes. Please check [the migration tutorial](https://mmediting.readthedocs.io/en/latest/migration/overview.html) for more details.
