# 变更日志

## v0.10.0 (12/08/2021).

**亮点**

1. 支持 LIIF-RDN (CVPR'2021)
2. 支持 BasicVSR++ (NTIRE'2021)

**新功能**

- Video SR datasets 支持加载文件列表 ([#423](https://github.com/open-mmlab/mmediting/pull/423))
- 支持 persistent worker ([#426](https://github.com/open-mmlab/mmediting/pull/426))
- 支持 LIIF-RDN ([#428](https://github.com/open-mmlab/mmediting/pull/428), [#440](https://github.com/open-mmlab/mmediting/pull/440))
- 支持 BasicVSR++ ([#451](https://github.com/open-mmlab/mmediting/pull/451), [#467](https://github.com/open-mmlab/mmediting/pull/467))
- 支持 mim ([#455](https://github.com/open-mmlab/mmediting/pull/455))

**Bug 修复**

- 修复了 stat.py 中的 bug ([#420](https://github.com/open-mmlab/mmediting/pull/420))
- 修复了 tensor2img 函数中的 astype 错误 ([#429](https://github.com/open-mmlab/mmediting/pull/429))
- 修复了当 pytorch >= 1.7 时由  torch.new_tensor 导致的 device 错误 ([#465](https://github.com/open-mmlab/mmediting/pull/465))
- 修复了 .mmedit/apis/train.py 中的 _non_dist_train ([#473](https://github.com/open-mmlab/mmediting/pull/473))
- 修复了多节点分布式测试函数 ([#478](https://github.com/open-mmlab/mmediting/pull/478))

**兼容性更新**

- 对 pytorch2onnx 重构了 LIIF  ([#425](https://github.com/open-mmlab/mmediting/pull/425))

**改进**

- 更新了部分中文文档 ([#415](https://github.com/open-mmlab/mmediting/pull/415), [#416](https://github.com/open-mmlab/mmediting/pull/416), [#418](https://github.com/open-mmlab/mmediting/pull/418), [#421](https://github.com/open-mmlab/mmediting/pull/421), [#424](https://github.com/open-mmlab/mmediting/pull/424), [#431](https://github.com/open-mmlab/mmediting/pull/431), [#442](https://github.com/open-mmlab/mmediting/pull/442))
- 添加了 pytorch 1.9.0 的 CI ([#444](https://github.com/open-mmlab/mmediting/pull/444))
- 重写了 README.md 的 configs 文件 ([#452](https://github.com/open-mmlab/mmediting/pull/452))
- 避免在单元测试中加载 VGG 网络的预训练权重 ([#466](https://github.com/open-mmlab/mmediting/pull/466))
- 支持在 div2k 数据集预处理时指定 scales ([#472](https://github.com/open-mmlab/mmediting/pull/472))
- 支持 readthedocs 中的所有格式 ([#479](https://github.com/open-mmlab/mmediting/pull/479))
- 使用 mmcv 的 version_info ([#480](https://github.com/open-mmlab/mmediting/pull/480))
- 删除了 restoration_video_demo.py 中不必要的代码 ([#484](https://github.com/open-mmlab/mmediting/pull/484))
- 将 DistEvalIterHook 的优先级修改为 'LOW' ([#489](https://github.com/open-mmlab/mmediting/pull/489))
- 重置资源限制 ([#491](https://github.com/open-mmlab/mmediting/pull/491))
- 在 README_CN.md 中更新了 QQ 的 QR code  ([#494](https://github.com/open-mmlab/mmediting/pull/494))
- 添加了 `myst_parser` ([#495](https://github.com/open-mmlab/mmediting/pull/495))
- 添加了 license 信息 ([#496](https://github.com/open-mmlab/mmediting/pull/496))
- 修正了 StyleGAN modules 中的拼写错误 ([#427](https://github.com/open-mmlab/mmediting/pull/427))
- 修正了 docs/demo.md 中的拼写错误 ([#453](https://github.com/open-mmlab/mmediting/pull/453), [#454](https://github.com/open-mmlab/mmediting/pull/454))
- 修复了 tools/data/super-resolution/reds/README.md 中的拼写错误 ([#469](https://github.com/open-mmlab/mmediting/pull/469))


## v0.9.0 (30/06/2021).

**主要更新**

1. 支持 DIC 和 DIC-GAN (CVPR'2020)
2. 支持 GLEAN Cat 8x (CVPR'2021)
3. 支持 TTSR-GAN (CVPR'2020)
4. 增加超分辨率 colab 使用指南

**新功能**

- 添加 DIC ([#342](https://github.com/open-mmlab/mmediting/pull/342), [#345](https://github.com/open-mmlab/mmediting/pull/345), [#348](https://github.com/open-mmlab/mmediting/pull/348), [#350](https://github.com/open-mmlab/mmediting/pull/350), [#351](https://github.com/open-mmlab/mmediting/pull/351), [#357](https://github.com/open-mmlab/mmediting/pull/357), [#363](https://github.com/open-mmlab/mmediting/pull/363), [#365](https://github.com/open-mmlab/mmediting/pull/365), [#366](https://github.com/open-mmlab/mmediting/pull/366))
- 增加 SRFolderMultipleGTDataset ([#355](https://github.com/open-mmlab/mmediting/pull/355))
- 增加 GLEAN Cat 8x ([#367](https://github.com/open-mmlab/mmediting/pull/367))
- 增加 SRFolderVideoDataset ([#370](https://github.com/open-mmlab/mmediting/pull/370))
- 增加超分辨率 colab 使用指南  ([#380](https://github.com/open-mmlab/mmediting/pull/380))
- 增加 TTSR-GAN ([#372](https://github.com/open-mmlab/mmediting/pull/372), [#381](https://github.com/open-mmlab/mmediting/pull/381), [#383](https://github.com/open-mmlab/mmediting/pull/383), [#398](https://github.com/open-mmlab/mmediting/pull/398))
- 增加 DIC-GAN ([#392](https://github.com/open-mmlab/mmediting/pull/392), [#393](https://github.com/open-mmlab/mmediting/pull/393), [#394](https://github.com/open-mmlab/mmediting/pull/394))

**Bug 修复**

- 修复了 restoration_video_inference.py 中的 bug ([#379](https://github.com/open-mmlab/mmediting/pull/379))
- 修复了 LIIF 的配置文件 ([#368](https://github.com/open-mmlab/mmediting/pull/368))
- 修改了 pre-trained EDVR-M 的路径 ([#396](https://github.com/open-mmlab/mmediting/pull/396))
- 修复了 restoration_video_inference 中的 normalization ([#406](https://github.com/open-mmlab/mmediting/pull/406))
- 修复了单元测试中的 [brush_stroke_mask] 错误 ([#409](https://github.com/open-mmlab/mmediting/pull/409))

**兼容性更新**

- 更改 mmcv 最低兼容版本为 v1.3 ([#378](https://github.com/open-mmlab/mmediting/pull/378))

**改进**

- 修正了代码中的拼写错误 ([#371](https://github.com/open-mmlab/mmediting/pull/371))
- 添加了 Custom_hooks ([#362](https://github.com/open-mmlab/mmediting/pull/362))
- 重构了 unittest 的目录结构 ([#386](https://github.com/open-mmlab/mmediting/pull/386))
- 添加了 Vid4 数据集的文档和下载链接 ([#399](https://github.com/open-mmlab/mmediting/pull/399))
- 更新了文档中的 model zoo ([#400](https://github.com/open-mmlab/mmediting/pull/400))
- 更新了 metafile ([407](https://github.com/open-mmlab/mmediting/pull/407))

## v0.8.0 (31/05/2021).

**主要更新**

1. 支持 GLEAN (CVPR'2021)
2. 支持 TTSR (CVPR'2020)
3. 支持 TDAN (CVPR'2020)

**新功能**

- 添加了 GLEAN ([#296](https://github.com/open-mmlab/mmediting/pull/296), [#332](https://github.com/open-mmlab/mmediting/pull/332))
- 支持 PWD metafile ([#298](https://github.com/open-mmlab/mmediting/pull/298))
- 支持 CropLike in pipeline ([#299](https://github.com/open-mmlab/mmediting/pull/299))
- 添加了 TTSR ([#301](https://github.com/open-mmlab/mmediting/pull/301), [#304](https://github.com/open-mmlab/mmediting/pull/304), [#307](https://github.com/open-mmlab/mmediting/pull/307), [#311](https://github.com/open-mmlab/mmediting/pull/311), [#311](https://github.com/open-mmlab/mmediting/pull/311), [#312](https://github.com/open-mmlab/mmediting/pull/312), [#313](https://github.com/open-mmlab/mmediting/pull/313), [#314](https://github.com/open-mmlab/mmediting/pull/314), [#321](https://github.com/open-mmlab/mmediting/pull/321), [#326](https://github.com/open-mmlab/mmediting/pull/326), [#327](https://github.com/open-mmlab/mmediting/pull/327))
- 添加了 TDAN ([#316](https://github.com/open-mmlab/mmediting/pull/316), [#334](https://github.com/open-mmlab/mmediting/pull/334), [#347](https://github.com/open-mmlab/mmediting/pull/347))
- 添加了 onnx2tensorrt ([#317](https://github.com/open-mmlab/mmediting/pull/317))
- 添加了 tensorrt evaluation ([#328](https://github.com/open-mmlab/mmediting/pull/328))
- 添加了 SRFacicalLandmarkDataset ([#329](https://github.com/open-mmlab/mmediting/pull/329))
- 添加了对 DIC 的 key point 辅助模型  ([#336](https://github.com/open-mmlab/mmediting/pull/336), [#341](https://github.com/open-mmlab/mmediting/pull/341))
- 添加了对视频超分辨率方法的演示 ([#275](https://github.com/open-mmlab/mmediting/pull/275))
- 添加了 SR Folder Ref Dataset ([#292](https://github.com/open-mmlab/mmediting/pull/292))
- 支持对视频超分辨率模型的 FLOPs 计算 ([#309](https://github.com/open-mmlab/mmediting/pull/309))

**Bug 修复**

- 修复了 find_unused_parameters in PyTorch 1.8 for BasicVSR ([#290](https://github.com/open-mmlab/mmediting/pull/290))
- 修复了 error in publish_model.py for pt>=1.6 ([#291](https://github.com/open-mmlab/mmediting/pull/291))
- 修复了 PSNR when input is uint8 ([#294](https://github.com/open-mmlab/mmediting/pull/294))

**改进**

- 支持 LoadImageFromFile 的 backend  ([#293](https://github.com/open-mmlab/mmediting/pull/293), [#303](https://github.com/open-mmlab/mmediting/pull/303))
- 更新了视频超分数据集的 `metric_average_mode` ([#319](https://github.com/open-mmlab/mmediting/pull/319))
- 添加了 restoration_demo.py 中的错误提示信息 ([324](https://github.com/open-mmlab/mmediting/pull/324))
- 修改了 getting_started.md ([#339](https://github.com/open-mmlab/mmediting/pull/339))
- 更新了 Vimeo90K 的描述 ([#349](https://github.com/open-mmlab/mmediting/pull/349))
- 支持在 GenerateSegmentIndices 中使用 start_index ([#338](https://github.com/open-mmlab/mmediting/pull/338))
- 支持在 GenerateSegmentIndices 使用不同的文件名模板 ([#325](https://github.com/open-mmlab/mmediting/pull/325))
- 支持使用 scale-factor 进行 resize ([#295](https://github.com/open-mmlab/mmediting/pull/295), [#310](https://github.com/open-mmlab/mmediting/pull/310))

## v0.7.0 (30/04/2021).

**主要更新**

1. 支持 BasicVSR (CVPR'2021)
2. 支持 IconVSR (CVPR'2021)
3. 支持 RDN (CVPR'2018)
4. 添加了 onnx evaluation 工具

**新功能**

- 添加了 RDN ([#233](https://github.com/open-mmlab/mmediting/pull/233), [#260](https://github.com/open-mmlab/mmediting/pull/260), [#280](https://github.com/open-mmlab/mmediting/pull/280))
- 添加了 MultipleGT 数据集 ([#238](https://github.com/open-mmlab/mmediting/pull/238))
- 添加了 BasicVSR and IconVSR ([#245](https://github.com/open-mmlab/mmediting/pull/245), [#252](https://github.com/open-mmlab/mmediting/pull/252), [#253](https://github.com/open-mmlab/mmediting/pull/253), [#254](https://github.com/open-mmlab/mmediting/pull/254), [#264](https://github.com/open-mmlab/mmediting/pull/264), [#274](https://github.com/open-mmlab/mmediting/pull/274), [#258](https://github.com/open-mmlab/mmediting/pull/258), [#252](https://github.com/open-mmlab/mmediting/pull/252), [#264](https://github.com/open-mmlab/mmediting/pull/264))
- 添加了 onnx evaluation 工具 ([#279](https://github.com/open-mmlab/mmediting/pull/279))

**Bug 修复**

- 修复了 maxunpool2d 的 onnx 转换  ([#243](https://github.com/open-mmlab/mmediting/pull/243))
- 修正了 `demo.md` 中的 inpainting ([#248](https://github.com/open-mmlab/mmediting/pull/248))
- 修正了 EDSR 的 config 文件 ([#251](https://github.com/open-mmlab/mmediting/pull/251))
- 修正了 README 中的链接 ([#256](https://github.com/open-mmlab/mmediting/pull/256))
- 修复了 restoration_inference 中 key missing 的 bug ([#270](https://github.com/open-mmlab/mmediting/pull/270))
- 修复了 channel_order 在 `loading.py` 的使用 ([#271](https://github.com/open-mmlab/mmediting/pull/271))
- 修复了 inpainting 的 command ([#278](https://github.com/open-mmlab/mmediting/pull/278))
- 修复了 `preprocess_vimeo90k_dataset.py` 中的 args 名称 ([#281](https://github.com/open-mmlab/mmediting/pull/281))

**改进**

- 支持 `test.py` 中的 `empty_cache` 选项  ([#261](https://github.com/open-mmlab/mmediting/pull/261))
- 更新了 README 中的 projects ([#249](https://github.com/open-mmlab/mmediting/pull/249), [#276](https://github.com/open-mmlab/mmediting/pull/276))
- 支持计算 Y-channel 的 PSNR and SSIM ([#250](https://github.com/open-mmlab/mmediting/pull/250))
- 添加了 zh-CN README ([#262](https://github.com/open-mmlab/mmediting/pull/262))
- 更新了 pytorch2onnx doc ([#265](https://github.com/open-mmlab/mmediting/pull/265))
- 删除了 README 中多余的引文  ([#268](https://github.com/open-mmlab/mmediting/pull/268))
- 更改了 tags 以 comment ([#269](https://github.com/open-mmlab/mmediting/pull/269))
- 在 README 中列出了 `model zoo` ([#284](https://github.com/open-mmlab/mmediting/pull/284), [#285](https://github.com/open-mmlab/mmediting/pull/285), [#286](https://github.com/open-mmlab/mmediting/pull/286))

## v0.6.0 (08/04/2021).

**主要更新**

1. 支持 Local Implicit Image Function (LIIF)
2. 支持将 DIM 和 GCA 从 Pytorch 导出到 ONNX

**新功能**

- 添加了 readthedocs 的配置文件以及修复了 docstring ([#92](https://github.com/open-mmlab/mmediting/pull/92))
- 添加了 github action file ([#94](https://github.com/open-mmlab/mmediting/pull/94))
- 支持将 DIM 和 GCA 从 Pytorch 导出到 ONNX  ([#105](https://github.com/open-mmlab/mmediting/pull/105))
- 支持 concatenating datasets ([#106](https://github.com/open-mmlab/mmediting/pull/106))
- 支持 `non_dist_train` validation ([#110](https://github.com/open-mmlab/mmediting/pull/110))
- 添加了 matting 的 colab 教程 ([#111](https://github.com/open-mmlab/mmediting/pull/111))
- 支持 niqe metric ([#114](https://github.com/open-mmlab/mmediting/pull/114))
- 对 parrots 支持 PoolDataLoader ([#134](https://github.com/open-mmlab/mmediting/pull/134))
- 支持 collect-env ([#137](https://github.com/open-mmlab/mmediting/pull/137), [#143](https://github.com/open-mmlab/mmediting/pull/143))
- 支持 CI 中的 pt1.6 cpu/gpu ([#138](https://github.com/open-mmlab/mmediting/pull/138))
- 支持 fp16 ([139](https://github.com/open-mmlab/mmediting/pull/139), [#144](https://github.com/open-mmlab/mmediting/pull/144))
- 支持发布到 pypi ([#149](https://github.com/open-mmlab/mmediting/pull/149))
- 添加了 modelzoo 的数据 ([#171](https://github.com/open-mmlab/mmediting/pull/171), [#182](https://github.com/open-mmlab/mmediting/pull/182), [#186](https://github.com/open-mmlab/mmediting/pull/186))
- 添加了数据集的文档 ([194](https://github.com/open-mmlab/mmediting/pull/194))
- 支持扩展 foreground 选项. ([#195](https://github.com/open-mmlab/mmediting/pull/195), [#199](https://github.com/open-mmlab/mmediting/pull/199), [#200](https://github.com/open-mmlab/mmediting/pull/200), [#210](https://github.com/open-mmlab/mmediting/pull/210))
- 支持 nn.MaxUnpool2d ([#196](https://github.com/open-mmlab/mmediting/pull/196))
- 添加了一些 FBA 组件 ([#203](https://github.com/open-mmlab/mmediting/pull/203), [#209](https://github.com/open-mmlab/mmediting/pull/209), [#215](https://github.com/open-mmlab/mmediting/pull/215), [#220](https://github.com/open-mmlab/mmediting/pull/220))
- 支持在数据预处理流水线中使用 random down sampling ([#222](https://github.com/open-mmlab/mmediting/pull/222))
- 支持 SR folder GT Dataset ([#223](https://github.com/open-mmlab/mmediting/pull/223))
- 支持 Local Implicit Image Function (LIIF) ([#224](https://github.com/open-mmlab/mmediting/pull/224), [#226](https://github.com/open-mmlab/mmediting/pull/226), [#227](https://github.com/open-mmlab/mmediting/pull/227), [#234](https://github.com/open-mmlab/mmediting/pull/234), [#239](https://github.com/open-mmlab/mmediting/pull/239))

**Bug 修复**
- 修复了 train api 中的 `_non_dist_train` ([#104](https://github.com/open-mmlab/mmediting/pull/104))
- 修复了 setup 和 CI ([#109](https://github.com/open-mmlab/mmediting/pull/109))
- 修复了 Normalize 中会导致多余循环的 bug ([#121](https://github.com/open-mmlab/mmediting/pull/121))
- 修复了 `get_hash` in `setup.py` ([#124](https://github.com/open-mmlab/mmediting/pull/124))
- 修复了 `tool/preprocess_reds_dataset.py` ([#148](https://github.com/open-mmlab/mmediting/pull/148))
- 修复了 `getting_started.md` 中的 slurm 训练教程 ([#162](https://github.com/open-mmlab/mmediting/pull/162))
- 修复了 pip 安装的 bug ([#173](https://github.com/open-mmlab/mmediting/pull/173))
- 修复了 config file 中的 bug ([#185](https://github.com/open-mmlab/mmediting/pull/185))
- 修复了数据集中失效的链接 ([#236](https://github.com/open-mmlab/mmediting/pull/236))
- 修复了 model zoo 中失效的链接 ([#242](https://github.com/open-mmlab/mmediting/pull/242))

**兼容性更新**

- 重构了 data loader 配置文件 ([#201](https://github.com/open-mmlab/mmediting/pull/201))

**改进**

- 更新了 requirements.txt ([#95](https://github.com/open-mmlab/mmediting/pull/95), [#100](https://github.com/open-mmlab/mmediting/pull/100))
- 更新了 teaser ([#96](https://github.com/open-mmlab/mmediting/pull/96))
- 更新了 README ([#93](https://github.com/open-mmlab/mmediting/pull/93), [#97](https://github.com/open-mmlab/mmediting/pull/97), [#98](https://github.com/open-mmlab/mmediting/pull/98), [#152](https://github.com/open-mmlab/mmediting/pull/152))
- 更新了 model_zoo ([#101](https://github.com/open-mmlab/mmediting/pull/101))
- 修正了一些 typos ([#102](https://github.com/open-mmlab/mmediting/pull/102), [#188](https://github.com/open-mmlab/mmediting/pull/188), [#191](https://github.com/open-mmlab/mmediting/pull/191), [#197](https://github.com/open-mmlab/mmediting/pull/197), [#208](https://github.com/open-mmlab/mmediting/pull/208))
- 采用了 skimage 中的 adjust_gamma 以及减少了依赖 ([#112](https://github.com/open-mmlab/mmediting/pull/112))
- 移除了 `.gitlab-ci.yml` ([#113](https://github.com/open-mmlab/mmediting/pull/113))
- 更新了第一方的代码库的引入 ([#115](https://github.com/open-mmlab/mmediting/pull/115))
- 移除了引用信息和联系方式 ([#122](https://github.com/open-mmlab/mmediting/pull/122))
- 更新了版本信息文件 ([#136](https://github.com/open-mmlab/mmediting/pull/136))
- 更新了下载链接 ([#141](https://github.com/open-mmlab/mmediting/pull/141))
- 更新了 `setup.py` ([#150](https://github.com/open-mmlab/mmediting/pull/150))
- 更新了所支持的 mmcv 的最高版本 ([#153](https://github.com/open-mmlab/mmediting/pull/153), [#154](https://github.com/open-mmlab/mmediting/pull/154))
- 更新了 `Crop` 以处理一系列来自视频的帧 ([#164](https://github.com/open-mmlab/mmediting/pull/164))
- 添加了其他 mm 项目的链接 ([#179](https://github.com/open-mmlab/mmediting/pull/179), [#180](https://github.com/open-mmlab/mmediting/pull/180))
- 添加了 config type ([#181](https://github.com/open-mmlab/mmediting/pull/181))
- 重写了文档 ([#184](https://github.com/open-mmlab/mmediting/pull/184))
- 添加了 config link ([#187](https://github.com/open-mmlab/mmediting/pull/187))
- 更新了文件结构 ([#192](https://github.com/open-mmlab/mmediting/pull/192))
- 更新了配置文档 ([#202](https://github.com/open-mmlab/mmediting/pull/202))
- 更新了 `slurm_train.md` 的脚本 ([#204](https://github.com/open-mmlab/mmediting/pull/204))
- 改进了代码风格 ([#206](https://github.com/open-mmlab/mmediting/pull/206), [#207](https://github.com/open-mmlab/mmediting/pull/207))
- 在 CompositeFg 使用了 `file_client`  ([#212](https://github.com/open-mmlab/mmediting/pull/212))
- 使用 `numpy.random` 代替了 `random` ([#213](https://github.com/open-mmlab/mmediting/pull/213))
- 重构了 `loader_cfg` ([#214](https://github.com/open-mmlab/mmediting/pull/214))

## v0.5.0 (09/07/2020).

请注意，作为MMEdit的一部分，**MMSR** 已经被合并到此代码库中。我们希望通过对新框架的精心设计和细致实现，MMEditing 能够为您提供更好的体验。
