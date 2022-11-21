# 变更日志

## v0.16.0 (31/10/2022)

**接口变更**

`VisualizationHook` 即将废弃，建议用户使用 `MMEditVisualizationHook`。(#1375)

<table align="center">
  <thead>
      <tr align='center'>
          <td>旧版本</td>
          <td>新版本</td>
      </tr>
  </thead>
  <tbody><tr valign='top'>
  <th>

```python
visual_config = dict(  # 构建可视化钩子的配置
  type='VisualizationHook',
  output_dir='visual',
  interval=1000,
  res_name_list=[
      'gt_img', 'masked_img', 'fake_res', 'fake_img', 'fake_gt_local'
  ],
)
```

</th>
  <th>

```python
visual_config = dict(  # 构建可视化钩子的配置
  type='MMEditVisualizationHook',
  output_dir='visual',
  interval=1000,
  res_name_list=[
      'gt_img', 'masked_img', 'fake_res', 'fake_img', 'fake_gt_local'
  ],
)
```

</th></tr>
</tbody></table>

**改进**

- 改进 `preprocess_div2k_dataset.py` 中的参数类型。 (#1381)
- 更新 RDN 的 docstring。 (#1326)
- 更新 `readme.md` 中的介绍说明。 (#)

**Bug 修复**

- 修复 FLAVR 在 `mmedit/models/video_interpolators` 中的注册和使用。(#1186)
- 修复 `restoration_video_inference.py` 中的路径处理问题。 (#1262)
- 修正 RDB 模型结构中的卷积通道数。(#1292, #1311)

**Contributors**

一共有 5 位 开发者对本次发布做出贡献。感谢 @LeoXing1996, @Z-Fran, @zengyh1900, @ryanxingql, @ruoningYu。

## v0.15.2 (09/09/2022)

**改进**

- \[Docs\] 修正文档中的拼写错误 by @Yulv-git in https://github.com/open-mmlab/mmediting/pull/1079
- \[Docs\] 修正 model zoo 数据集的链接 by @Z-Fran in https://github.com/open-mmlab/mmediting/pull/1043
- \[Docs\] 修正 readme 中的拼写错误 by @arch-user-france1 in https://github.com/open-mmlab/mmediting/pull/1078
- \[Improve\] 提供 FLAVR demo by @Yshuo-Li in https://github.com/open-mmlab/mmediting/pull/954
- \[Fix\] 更新 MMCV 的版本上限到 1.7 by @wangruohui in https://github.com/open-mmlab/mmediting/pull/1001
- \[Improve\] 改进 niqe_pris_params.npz 安装路径 by @ychfan in https://github.com/open-mmlab/mmediting/pull/995
- \[CI\] 更新 Github Actions， CircleCI 以及 Issue 和 PR 的模板 by @zengyh1900 in https://github.com/open-mmlab/mmediting/pull/1087

**Contributors**

@wangruohui @Yshuo-Li @zengyh1900 @Z-Fran @ychfan @arch-user-france1 @Yulv-git

## v0.15.1 (04/07/2022)

**Bug 修复**

- \[修复\] 更新 cain_b5_g1b32_vimeo90k_triplet.py 配置文件 ([#929](https://github.com/open-mmlab/mmediting/pull/929))
- \[文档\] 修复 OST 数据集的链接 ([#933](https://github.com/open-mmlab/mmediting/pull/933))

**改进**

- \[文档\] 更新 OST 数据集指令 ([#937](https://github.com/open-mmlab/mmediting/pull/937))
- \[测试\] 在 CUDA 环境中没有实际执行 ([#921](https://github.com/open-mmlab/mmediting/pull/921))
- \[文档\] 首页演示视频添加水印 ([#935](https://github.com/open-mmlab/mmediting/pull/935))
- \[测试\] 添加 mim ci ([#928](https://github.com/open-mmlab/mmediting/pull/928))
- \[文档\] 更新 FLAVR 的 README.md ([#919](https://github.com/open-mmlab/mmediting/pull/919))
- \[改进\] 更新 .pre-commit-config.yaml 中的 md-format ([#917](https://github.com/open-mmlab/mmediting/pull/917))
- \[改进\] 在 setup.py 中添加 miminstall.txt ([#916](https://github.com/open-mmlab/mmediting/pull/916))
- \[修复\] 修复 dim/README.md 中的混乱问题 ([#913](https://github.com/open-mmlab/mmediting/pull/913))
- \[改进\] 跳过有问题的 opencv-python 版本 ([#833](https://github.com/open-mmlab/mmediting/pull/833))

**贡献者**

@wangruohui @Yshuo-Li

## v0.15.0 (01/06/2022)

**Highlights主要更新**

1. 支持 FLAVR
2. 支持 AOT-GAN
3. 在 CAIN 中支持 ReduceLROnPlateau 策略

**新功能**

- 支持 AOT-GAN ([#681](https://github.com/open-mmlab/mmediting/pull/681))
- 支持 Vimeo90k-triplet 数据集 ([#810](https://github.com/open-mmlab/mmediting/pull/810))
- 为 mm-assistant 添加默认 config ([#827](https://github.com/open-mmlab/mmediting/pull/827))
- 支持 CPU demo ([#848](https://github.com/open-mmlab/mmediting/pull/848))
- 在 `LoadImageFromFileList` 中支持 `use_cache` 和 `backend` ([#857](https://github.com/open-mmlab/mmediting/pull/857))
- 支持 VFIVimeo90K7FramesDataset ([#858](https://github.com/open-mmlab/mmediting/pull/858))
- 在 VFI pipeline 中支持 ColorJitter ([#859](https://github.com/open-mmlab/mmediting/pull/859))
- 支持 ReduceLrUpdaterHook ([#860](https://github.com/open-mmlab/mmediting/pull/860))
- 在 IterBaseRunner 中支持 `after_val_epoch` ([#861](https://github.com/open-mmlab/mmediting/pull/861))
- 支持 FLAVR Net ([#866](https://github.com/open-mmlab/mmediting/pull/866), [#867](https://github.com/open-mmlab/mmediting/pull/867), [#897](https://github.com/open-mmlab/mmediting/pull/897))
- 支持 MAE 评估方式 ([#871](https://github.com/open-mmlab/mmediting/pull/871))
- 使用 mdformat ([#888](https://github.com/open-mmlab/mmediting/pull/888))
- 在 CAIN 中支持 ReduceLROnPlateau 策略 ([#906](https://github.com/open-mmlab/mmediting/pull/906))

**Bug 修复**

- 在 restoration_demo.py 中将 `-` 改为 `_` ([#834](https://github.com/open-mmlab/mmediting/pull/834))
- 移除 requirements/docs.txt 中的 recommonmark ([#844](https://github.com/open-mmlab/mmediting/pull/844))
- 将 README 中的 EDVR 移动到 VSR 类别中 ([#849](https://github.com/open-mmlab/mmediting/pull/849))
- 修改 crop.py，移除跨栏 F-string 中的 `,` ([#855](https://github.com/open-mmlab/mmediting/pull/855))
- 修改 test_pipeline，将重复的 `lq_path` 改为 `gt_path` ([#862](https://github.com/open-mmlab/mmediting/pull/862))
- 修复 TOF-VFI 的 unittest 问题 ([#873](https://github.com/open-mmlab/mmediting/pull/873))
- 解决 VFI demo 中帧序列出错问题 ([#891](https://github.com/open-mmlab/mmediting/pull/891))
- 修复 README 中的 logo & contrib 链接 ([#898](https://github.com/open-mmlab/mmediting/pull/898))
- 修复 indexnet_dimaug_mobv2_1x16_78k_comp1k.py ([#901](https://github.com/open-mmlab/mmediting/pull/901))

**改进**

- 在训练和测试脚本中增加 `--cfg-options` 参数 ([#826](https://github.com/open-mmlab/mmediting/pull/826))
- 更新 MMCV_MAX 到 1.6 ([#829](https://github.com/open-mmlab/mmediting/pull/829))
- 在 README 中更新 TOFlow ([#835](https://github.com/open-mmlab/mmediting/pull/835))
- 恢复 beirf 安装步骤，合并可选要求 ([#836](https://github.com/open-mmlab/mmediting/pull/836))
- 在 citation 中使用 {MMEditing Contributors} ([#838](https://github.com/open-mmlab/mmediting/pull/838))
- 增加定制损失函数的教程 ([#839](https://github.com/open-mmlab/mmediting/pull/839))
- 在 README 中添加安装指南 (wiki ver) ([#845](https://github.com/open-mmlab/mmediting/pull/845))
- 在中文文档中添加“需要帮助翻译”的说明 ([#850](https://github.com/open-mmlab/mmediting/pull/850))
- 在 README_zh-CN.md 中添加微信二维码 ([#851](https://github.com/open-mmlab/mmediting/pull/851))
- 支持 SRFolderVideoDataset 的非零帧索引，修复拼写错误 ([#853](https://github.com/open-mmlab/mmediting/pull/853))
- 创建 docker 的 README.md ([#856](https://github.com/open-mmlab/mmediting/pull/856))
- 优化 IO 流量偏差 ([#881](https://github.com/open-mmlab/mmediting/pull/881))
- 将 wiki/installation 移到 docs ([#883](https://github.com/open-mmlab/mmediting/pull/883))
- 添加 `myst_heading_anchors` ([#887](https://github.com/open-mmlab/mmediting/pull/887))
- 在 inpainting demo 中使用预训练模型链接 ([#892](https://github.com/open-mmlab/mmediting/pull/892))

**贡献者**

@wangruohui @quincylin1 @nijkah @jayagami @ckkelvinchan @ryanxingql @NK-CS-ZZL @Yshuo-Li

## v0.14.0 (01/04/2022)

**Highlights主要更新**

1. 支持视频插帧算法 TOFlow

**新功能**

- 支持 AOT-GAN ([#677](https://github.com/open-mmlab/mmediting/pull/677))
- 使用 `--diff-seed` 在多卡训练中为 torch 设置不同的初始化种子 ([#781](https://github.com/open-mmlab/mmediting/pull/781))
- 在视频插帧 demo 中支持流帧读取 ([#790](https://github.com/open-mmlab/mmediting/pull/790))
- 支持非 slurm 的 dist_train ([#791](https://github.com/open-mmlab/mmediting/pull/791))
- 在 restoration_video_demo 中将 LQ 存放在 CPU ([#792](https://github.com/open-mmlab/mmediting/pull/792))
- 在 EDSR 中支持处理灰度数据 ([#793](https://github.com/open-mmlab/mmediting/pull/793))
- 支持视频插帧算法 TOFlow ([#806](https://github.com/open-mmlab/mmediting/pull/806), [#811](https://github.com/open-mmlab/mmediting/pull/811))
- 支持在 DistributedSampler 中为不同的 rank 设置不同的随机种子 ([#815](https://github.com/open-mmlab/mmediting/pull/815))

**Bug 修复**

- 修复 README 文件中的失效链接 ([#782](https://github.com/open-mmlab/mmediting/pull/782), [#786](https://github.com/open-mmlab/mmediting/pull/786), [#819](https://github.com/open-mmlab/mmediting/pull/819), [#820](https://github.com/open-mmlab/mmediting/pull/820))
- 修复抠图教程 ([#795](https://github.com/open-mmlab/mmediting/pull/795))
- 翻转 `RandomAffine` 中的 `flip_ratio` ([#799](https://github.com/open-mmlab/mmediting/pull/799))
- 修复 preprocess_div2k_dataset.py ([#801](https://github.com/open-mmlab/mmediting/pull/801))
- 修复 SR Colab Demo Installation 方法和 Set5 链接 ([#807](https://github.com/open-mmlab/mmediting/pull/807))
- 修正EDSR README 中的 Y/GRB 错误 ([#812](https://github.com/open-mmlab/mmediting/pull/812))
- 将 `README(_zh-CN).md` 中的 pytorch 安装命令替换为 conda ([#816](https://github.com/open-mmlab/mmediting/pull/816))

**改进**

- 更新 CI ([#650](https://github.com/open-mmlab/mmediting/pull/650))
- 更新 requirements.txt ([#725](https://github.com/open-mmlab/mmediting/pull/725), [#817](https://github.com/open-mmlab/mmediting/pull/817))
- 增加 dataset ([#758](https://github.com/open-mmlab/mmediting/pull/758)), pipeline ([#779](https://github.com/open-mmlab/mmediting/pull/779)), model ([#766](https://github.com/open-mmlab/mmediting/pull/758)) 教程
- 更新 index 和 TOC 结构树 ([#767](https://github.com/open-mmlab/mmediting/pull/767))
- 更新 update_model_index.py 以兼容 Windows ([#768](https://github.com/open-mmlab/mmediting/pull/768))
- 更新文档构建系统 ([#769](https://github.com/open-mmlab/mmediting/pull/769))
- 更新 setuptools 的关键字和分类器 ([#773](https://github.com/open-mmlab/mmediting/pull/773))
- 更新安装文档 ([#776](https://github.com/open-mmlab/mmediting/pull/776), [#800](https://github.com/open-mmlab/mmediting/pull/800))
- 更新 BasicVSR++ 和 RealBasicVSR 文档 ([#778](https://github.com/open-mmlab/mmediting/pull/778))
- 更新 citation ([#785](https://github.com/open-mmlab/mmediting/pull/785), [#787](https://github.com/open-mmlab/mmediting/pull/787))
- 重组文档 ([#788](https://github.com/open-mmlab/mmediting/pull/788))
- 在 metafile 中使用 config 的全名作为 'Name'，以支持 mim 下载 ([#798](https://github.com/open-mmlab/mmediting/pull/798))
- 更新 README 中的图片和视频示例 ([#802](https://github.com/open-mmlab/mmediting/pull/802))
- 在视频插帧测试时使用 `clamp(0, 1)` ([#805](https://github.com/open-mmlab/mmediting/pull/805))
- 在演示和工具中的命令行参数中使用连字符 ([#808](https://github.com/open-mmlab/mmediting/pull/808)), 在python文件中为必选参数保留下划线 ([#822](https://github.com/open-mmlab/mmediting/pull/822))
- 将 `dataset.pipeline` 作为文档的单列内容 ([#813](https://github.com/open-mmlab/mmediting/pull/813))
- 更新 mmcv-full>=1.3.13 以在 CPU 中支持 DCN ([#823](https://github.com/open-mmlab/mmediting/pull/823))

**贡献者**

@wangruohui @ckkelvinchan @Yshuo-Li @nijkah @wdmwhh @freepoet @quincylin1

## v0.13.0 (01/03/2022)

**Highlights主要更新**

1. 支持 CAIN
2. 支持 EDVR-L
3. 支持在 Windows 系统中运行

**New Features**

- 为图像和视频添加测试时间 ensemble，并支持 BasicVSR 系列中的 ensemble ([#585](https://github.com/open-mmlab/mmediting/pull/585))
- 支持 AOT-GAN (正在进行中的工作) ([#674](https://github.com/open-mmlab/mmediting/pull/674), [#675](https://github.com/open-mmlab/mmediting/pull/675), [#676](https://github.com/open-mmlab/mmediting/pull/676))
- 支持 CAIN ([#683](https://github.com/open-mmlab/mmediting/pull/683), [#691](https://github.com/open-mmlab/mmediting/pull/691), [#709](https://github.com/open-mmlab/mmediting/pull/709), [#713](https://github.com/open-mmlab/mmediting/pull/713))
- 新增 basic interpolater ([#687](https://github.com/open-mmlab/mmediting/pull/687))
- 新增 BaseVFIDataset and VFIVimeo90KDataset ([#695](https://github.com/open-mmlab/mmediting/pull/695), [#697](https://github.com/open-mmlab/mmediting/pull/697))
- 新增 video interpolation demo ([#688](https://github.com/open-mmlab/mmediting/pull/688), [#717](https://github.com/open-mmlab/mmediting/pull/717))
- 在 RDDBNet 中支持多种 scale ([#699](https://github.com/open-mmlab/mmediting/pull/699))
- 在 demo 中支持 Ref-SR 推理 ([#716](https://github.com/open-mmlab/mmediting/pull/716))
- 在 REDS 数据集上支持 EDVR-L ([#719](https://github.com/open-mmlab/mmediting/pull/719))
- 支持 CPU 训练 ([#720](https://github.com/open-mmlab/mmediting/pull/720))
- 支持在 Windows 中运行 ([#732](https://github.com/open-mmlab/mmediting/pull/732), [#738](https://github.com/open-mmlab/mmediting/pull/738))
- 支持 CPU 中的 DCN ([#735](https://github.com/open-mmlab/mmediting/pull/735))

**Bug 修复**

- 修复文档中的链接问题 ([#703](https://github.com/open-mmlab/mmediting/pull/703), [#704](https://github.com/open-mmlab/mmediting/pull/704))
- 修复 Dockerfile 中的 `MMCV` 参数 ([#708](https://github.com/open-mmlab/mmediting/pull/708))
- 修复不可执行文件的文件权限 ([#718](https://github.com/open-mmlab/mmediting/pull/718))
- 修复一些与 numpy 相关的弃用警告 ([#728](https://github.com/open-mmlab/mmediting/pull/728))
- 删除 `TestVFIDataset` 中的 `__init__` ([#731](https://github.com/open-mmlab/mmediting/pull/731))
- 修复数据集说明文档中的数据类型 ([#739](https://github.com/open-mmlab/mmediting/pull/739))
- 修复说明文档中的数学符号 ([#741](https://github.com/open-mmlab/mmediting/pull/741))
- 修复 copyright commit hook 中忽略的文件夹 ([#754](https://github.com/open-mmlab/mmediting/pull/754))
- 删除加载中的重复测试 ([#756](https://github.com/open-mmlab/mmediting/pull/756))

**改进**

- 将 CI 中的 Pillow 版本从 6.2.2 to更新至 8.4 ([#693](https://github.com/open-mmlab/mmediting/pull/693))
- 在 SRREDSMultipleGTDataset 中增加 'repeat' 参数 ([#672](https://github.com/open-mmlab/mmediting/pull/672))
- 弃用对 "python setup.py test" 的支持 ([#701](https://github.com/open-mmlab/mmediting/pull/701))
- 在训练和测试中添加 `multi-processing` 设置 ([#707](https://github.com/open-mmlab/mmediting/pull/707))
- 添加 OpenMMLab 网站和平台链接 ([#710](https://github.com/open-mmlab/mmediting/pull/710))
- 重构各模型的 README 文件 ([#712](https://github.com/open-mmlab/mmediting/pull/712))
- 使用 `package.version.parse` 替代字符串版本比较 ([#723](https://github.com/open-mmlab/mmediting/pull/723))
- 添加 Ref-SR 演示和视频帧插值演示的文档 ([#724](https://github.com/open-mmlab/mmediting/pull/724))
- 重构 README.md 并增加插帧算法相关内容 ([#726](https://github.com/open-mmlab/mmediting/pull/726))
- 更新 pre-commit hook 中的 isort 版本 ([#727](https://github.com/open-mmlab/mmediting/pull/727))
- 重新设计 Linux 的 CI ([#734](https://github.com/open-mmlab/mmediting/pull/734))
- 更新 install.md ([#763](https://github.com/open-mmlab/mmediting/pull/763))
- 在 README 文件中重新组织 OpenMMLab 项目 ([#764](https://github.com/open-mmlab/mmediting/pull/764))
- 为部署工具添加弃用消息 ([#765](https://github.com/open-mmlab/mmediting/pull/765))

**贡献者**

@wangruohui @ckkelvinchan @Yshuo-Li @quincylin1 @Juggernaut93 @anse3832 @nijkah

## v0.12.0 (31/12/2021)

**主要更新**

1. 支持 RealBasicVSR
2. 支持 Real-ESRGAN 预训练模型

**新功能**

- 支持视频恢复演示中的视频输入和输出 ([#622](https://github.com/open-mmlab/mmediting/pull/622))
- 支持 RealBasicVSR ([#632](https://github.com/open-mmlab/mmediting/pull/632), [#633](https://github.com/open-mmlab/mmediting/pull/633), [#647](https://github.com/open-mmlab/mmediting/pull/647), [#680](https://github.com/open-mmlab/mmediting/pull/680))
- 支持 Real-ESRGAN 预训练模型 ([#635](https://github.com/open-mmlab/mmediting/pull/635))
- 加载图片时支持转化到 Y 通道 ([643](https://github.com/open-mmlab/mmediting/pull/643))
- 训练时支持随机视频压缩 ([#646](https://github.com/open-mmlab/mmediting/pull/646))
- 支持裁剪序列 ([#648](https://github.com/open-mmlab/mmediting/pull/648))
- 支持 pixel_unshuffle ([#684](https://github.com/open-mmlab/mmediting/pull/684))

**Bug 修复**

- 将 RandomResize 的 'target_size' 从列表更改为元组 ([#617](https://github.com/open-mmlab/mmediting/pull/617))
- 修复 preprocess_df2k_ost_dataset.py 中的文件夹创建问题 ([#623](https://github.com/open-mmlab/mmediting/pull/623))
- 在 README 中更改 TDAN 配置路径 ([#625](https://github.com/open-mmlab/mmediting/pull/625))
- 在 Real-ESRNet 配置中将 UnsharpMasking 的 'radius' 更改为 'kernel_size' ([#626](https://github.com/open-mmlab/mmediting/pull/626))
- 修复 MATLABLikeResize 中的 Bug ([#630](https://github.com/open-mmlab/mmediting/pull/630))
- 修复 'flow_warp' 注释 ([#655](https://github.com/open-mmlab/mmediting/pull/655))
- 修复文档中 Model Zoo 和 Datasets 的错误 ([#664](https://github.com/open-mmlab/mmediting/pull/664))
- 修复 'random_degradations' 中的错误 ([#673](https://github.com/open-mmlab/mmediting/pull/673))
- 限制 opencv-python 版本 ([#689](https://github.com/open-mmlab/mmediting/pull/689))

**改进**

- 将文档翻译成中文 ([#576](https://github.com/open-mmlab/mmediting/pull/576), [#577](https://github.com/open-mmlab/mmediting/pull/577), [#578](https://github.com/open-mmlab/mmediting/pull/578), [#579](https://github.com/open-mmlab/mmediting/pull/579), [#581](https://github.com/open-mmlab/mmediting/pull/581), [#582](https://github.com/open-mmlab/mmediting/pull/582), [#584](https://github.com/open-mmlab/mmediting/pull/584), [#587](https://github.com/open-mmlab/mmediting/pull/587), [#588](https://github.com/open-mmlab/mmediting/pull/588), [#589](https://github.com/open-mmlab/mmediting/pull/589), [#590](https://github.com/open-mmlab/mmediting/pull/590), [#591](https://github.com/open-mmlab/mmediting/pull/591), [#592](https://github.com/open-mmlab/mmediting/pull/592), [#593](https://github.com/open-mmlab/mmediting/pull/593), [#594](https://github.com/open-mmlab/mmediting/pull/594), [#595](https://github.com/open-mmlab/mmediting/pull/595), [#596](https://github.com/open-mmlab/mmediting/pull/596), [#641](https://github.com/open-mmlab/mmediting/pull/641), [#647](https://github.com/open-mmlab/mmediting/pull/647), [#656](https://github.com/open-mmlab/mmediting/pull/656), [#665](https://github.com/open-mmlab/mmediting/pull/665), [#666](https://github.com/open-mmlab/mmediting/pull/666))
- 添加 UNetDiscriminatorWithSpectralNorm ([#605](https://github.com/open-mmlab/mmediting/pull/605))
- 使用 PyTorch sphinx 主题 ([#607](https://github.com/open-mmlab/mmediting/pull/607), [#608](https://github.com/open-mmlab/mmediting/pull/608))
- 在文本文档中更新 mmcv ([#609](https://github.com/open-mmlab/mmediting/pull/609)), mmflow ([#621](https://github.com/open-mmlab/mmediting/pull/621)), mmfewshot ([#634](https://github.com/open-mmlab/mmediting/pull/634)) and mmhuman3d ([#649](https://github.com/open-mmlab/mmediting/pull/649)) 的信息
- 将最低 GCC 版本转换为 5.4 ([#612](https://github.com/open-mmlab/mmediting/pull/612))
- 在 SRDataset IMG_EXTENSIONS 中添加 tiff ([#614](https://github.com/open-mmlab/mmediting/pull/614))
- 更新 metafile 和 update_model_index.py ([#615](https://github.com/open-mmlab/mmediting/pull/615))
- 更新 preprocess_df2k_ost_dataset.py ([#624](https://github.com/open-mmlab/mmediting/pull/624))
- 将摘要添加到README ([#628](https://github.com/open-mmlab/mmediting/pull/628), [#636](https://github.com/open-mmlab/mmediting/pull/636))
- 将 NIQE 与 MATLAB 结果对齐 ([#631](https://github.com/open-mmlab/mmediting/pull/631))
- 添加官方 Markdown lint 钩子 ([#639](https://github.com/open-mmlab/mmediting/pull/639))
- 更改某些特定文件时跳过 CI ([#640](https://github.com/open-mmlab/mmediting/pull/640))
- 更新 docs/conf.py ([#644](https://github.com/open-mmlab/mmediting/pull/644), [#651](https://github.com/open-mmlab/mmediting/pull/651))
- 尝试在 Windows 上创建软链接 ([#645](https://github.com/open-mmlab/mmediting/pull/645))
- 取消之前未完成的运行 ([#650](https://github.com/open-mmlab/mmediting/pull/650))
- 更新 demo.md 和 getting_started.md 中 config 的路径 ([#658](https://github.com/open-mmlab/mmediting/pull/658), [#659](https://github.com/open-mmlab/mmediting/pull/659))
- 使用 mmcv 根模型注册表 ([#660](https://github.com/open-mmlab/mmediting/pull/660))
- 更新 README.md ([#654](https://github.com/open-mmlab/mmediting/pull/654), [#663](https://github.com/open-mmlab/mmediting/pull/663))
- 重构文档结构 ([#668](https://github.com/open-mmlab/mmediting/pull/668))
- 添加脚本以将 REDS 图像裁剪为子图像以加快 IO ([#669](https://github.com/open-mmlab/mmediting/pull/669))
- 将 metafile 中任务名称的第一个字母大写 ([#678](https://github.com/open-mmlab/mmediting/pull/678))
- 更新 FixedCrop 以支持裁剪图像序列 ([#682](https://github.com/open-mmlab/mmediting/pull/682))

## v0.11.0 (03/11/2021)

**亮点**

1. 支持使用 GLEAN 处理人脸图像的盲超分辨率
2. 支持 Real-ESRGAN 模型 #546

**新功能**

- 指数移动平均线钩子 #542
- 支持 DF2K_OST 数据 #566

**改进**

- 增加与 MATLAB 相似的双线性插值算法 #507
- 在训练期间支持随机退化 #504
- 支持 torchserve #568

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
- 修复了 .mmedit/apis/train.py 中的 \_non_dist_train ([#473](https://github.com/open-mmlab/mmediting/pull/473))
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
- 修复了单元测试中的 \[brush_stroke_mask\] 错误 ([#409](https://github.com/open-mmlab/mmediting/pull/409))

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
