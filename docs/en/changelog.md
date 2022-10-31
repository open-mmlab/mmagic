# Changelog

## v0.16.0 (31/10/2022)

**Deprecations**

`VisualizationHook` is deprecated. Users should use `MMEditVisualizationHook` instead. (#1375)

<table align="center">
  <thead>
      <tr align='center'>
          <td>Old Version</td>
          <td>Current Version</td>
      </tr>
  </thead>
  <tbody><tr valign='top'>
  <th>

```python
visual_config = dict(  # config to register visualization hook
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
visual_config = dict(  # config to register visualization hook
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

**New Features & Improvements**

- Improve arguments type in `preprocess_div2k_dataset.py`. (#1381)
- Update docstring of RDN. (#1326)
- Update the introduction in readme. (#)

**Bug Fixes**

- Fix FLAVR register in `mmedit/models/video_interpolators` when importing `FLAVR`. (#1186)
- Fix data path processing in `restoration_video_inference.py`. (#1262)
- Fix the number of channels in RDB. (#1292, #1311)

**Contributors**

A total of 5 developers contributed to this release.
Thanks @LeoXing1996, @Z-Fran, @zengyh1900, @ryanxingql, @ruoningYu.

## v0.15.2 (09/09/2022)

**Improvements**

- \[Docs\] Fix typos in docs. by @Yulv-git in https://github.com/open-mmlab/mmediting/pull/1079
- \[Docs\]  fix model_zoo and datasets docs link by @Z-Fran in https://github.com/open-mmlab/mmediting/pull/1043
- \[Docs\]  fix typos in readme. by @arch-user-france1 in https://github.com/open-mmlab/mmediting/pull/1078
- \[Improve\] FLAVR demo by @Yshuo-Li in https://github.com/open-mmlab/mmediting/pull/954
- \[Fix\] Update MMCV_MAX to 1.7 by @wangruohui in https://github.com/open-mmlab/mmediting/pull/1001
- \[Improve\] Fix niqe_pris_params.npz path when installed as package by @ychfan in https://github.com/open-mmlab/mmediting/pull/995
- \[CI\] update github workflow, circleci and github templates by @zengyh1900 in https://github.com/open-mmlab/mmediting/pull/1087

**Contributors**

@wangruohui @Yshuo-Li @zengyh1900 @Z-Fran @ychfan @arch-user-france1 @Yulv-git

## v0.15.1 (04/07/2022)

**Bug Fixes**

- \[Fix\] Update cain_b5_g1b32_vimeo90k_triplet.py ([#929](https://github.com/open-mmlab/mmediting/pull/929))
- \[Docs\] Fix link to OST dataset ([#933](https://github.com/open-mmlab/mmediting/pull/933))

**Improvements**

- \[Docs\] Update instruction to OST dataset ([#937](https://github.com/open-mmlab/mmediting/pull/937))
- \[CI\] No actual execution in CUDA envs ([#921](https://github.com/open-mmlab/mmediting/pull/921))
- \[Docs\] Add watermark to demo video ([#935](https://github.com/open-mmlab/mmediting/pull/935))
- \[Tests\] Add mim ci ([#928](https://github.com/open-mmlab/mmediting/pull/928))
- \[Docs\] Update README.md of FLAVR ([#919](https://github.com/open-mmlab/mmediting/pull/919))
- \[Improve\] Update md-format in .pre-commit-config.yaml ([#917](https://github.com/open-mmlab/mmediting/pull/917))
- \[Improve\] Add miminstall.txt in setup.py ([#916](https://github.com/open-mmlab/mmediting/pull/916))
- \[Fix\] Fix clutter in dim/README.md ([#913](https://github.com/open-mmlab/mmediting/pull/913))
- \[Improve\] Skip problematic opencv-python versions ([#833](https://github.com/open-mmlab/mmediting/pull/833))

**Contributors**

@wangruohui @Yshuo-Li

## v0.15.0 (01/06/2022)

**Highlights**

1. Support FLAVR
2. Support AOT-GAN
3. Support CAIN with ReduceLROnPlateau Scheduler

**New Features**

- Add configs for AOT-GAN ([#681](https://github.com/open-mmlab/mmediting/pull/681))
- Support Vimeo90k-triplet dataset ([#810](https://github.com/open-mmlab/mmediting/pull/810))
- Add default config for mm-assistant ([#827](https://github.com/open-mmlab/mmediting/pull/827))
- Support CPU demo ([#848](https://github.com/open-mmlab/mmediting/pull/848))
- Support `use_cache` and `backend` in LoadImageFromFileList ([#857](https://github.com/open-mmlab/mmediting/pull/857))
- Support VFIVimeo90K7FramesDataset ([#858](https://github.com/open-mmlab/mmediting/pull/858))
- Support ColorJitter for VFI ([#859](https://github.com/open-mmlab/mmediting/pull/859))
- Support ReduceLrUpdaterHook ([#860](https://github.com/open-mmlab/mmediting/pull/860))
- Support `after_val_epoch` in IterBaseRunner ([#861](https://github.com/open-mmlab/mmediting/pull/861))
- Support FLAVR Net ([#866](https://github.com/open-mmlab/mmediting/pull/866), [#867](https://github.com/open-mmlab/mmediting/pull/867), [#897](https://github.com/open-mmlab/mmediting/pull/897))
- Support MAE metric ([#871](https://github.com/open-mmlab/mmediting/pull/871))
- Use mdformat ([#888](https://github.com/open-mmlab/mmediting/pull/888))
- Support CAIN with ReduceLROnPlateau Scheduler ([#906](https://github.com/open-mmlab/mmediting/pull/906))

**Bug Fixes**

- Change `-` to `_` for restoration_demo.py ([#834](https://github.com/open-mmlab/mmediting/pull/834))
- Remove recommonmark in requirements/docs.txt ([#844](https://github.com/open-mmlab/mmediting/pull/844))
- Move EDVR to VSR category in README.md ([#849](https://github.com/open-mmlab/mmediting/pull/849))
- Remove `,` in multi-line F-string in crop.py ([#855](https://github.com/open-mmlab/mmediting/pull/855))
- Modify double `lq_path` to `gt_path` in test_pipeline ([#862](https://github.com/open-mmlab/mmediting/pull/862))
- Fix unittest of TOF-VFI ([#873](https://github.com/open-mmlab/mmediting/pull/873))
- Fix wrong frames in VFI demo ([#891](https://github.com/open-mmlab/mmediting/pull/891))
- Fix logo & contrib guideline on README ([#898](https://github.com/open-mmlab/mmediting/pull/898))
- Normalizing trimap in indexnet_dimaug_mobv2_1x16_78k_comp1k.py ([#901](https://github.com/open-mmlab/mmediting/pull/901))

**Improvements**

- Add `--cfg-options` in train/test scripts ([#826](https://github.com/open-mmlab/mmediting/pull/826))
- Update MMCV_MAX to 1.6 ([#829](https://github.com/open-mmlab/mmediting/pull/829))
- Update TOFlow in README ([#835](https://github.com/open-mmlab/mmediting/pull/835))
- Recover beirf installation steps & merge optional requirements ([#836](https://github.com/open-mmlab/mmediting/pull/836))
- Use {MMEditing Contributors} in citation ([#838](https://github.com/open-mmlab/mmediting/pull/838))
- Add tutorial for customizing losses ([#839](https://github.com/open-mmlab/mmediting/pull/839))
- Add installation guide (wiki ver) in README ([#845](https://github.com/open-mmlab/mmediting/pull/845))
- Add a 'need help to traslate' note on Chinese documentation ([#850](https://github.com/open-mmlab/mmediting/pull/850))
- Add wechat QR code in README_zh-CN.md ([#851](https://github.com/open-mmlab/mmediting/pull/851))
- Support non-zero frame index for SRFolderVideoDataset & Fix Typos ([#853](https://github.com/open-mmlab/mmediting/pull/853))
- Create README.md for docker ([#856](https://github.com/open-mmlab/mmediting/pull/856))
- Optimize IO for flow_warp ([#881](https://github.com/open-mmlab/mmediting/pull/881))
- Move wiki/installation to docs ([#883](https://github.com/open-mmlab/mmediting/pull/883))
- Add `myst_heading_anchors` ([#887](https://github.com/open-mmlab/mmediting/pull/887))
- Use checkpoint link in inpainting demo ([#892](https://github.com/open-mmlab/mmediting/pull/892))

**Contributors**

@wangruohui @quincylin1 @nijkah @jayagami @ckkelvinchan @ryanxingql @NK-CS-ZZL @Yshuo-Li

## v0.14.0 (01/04/2022)

**Highlights**

1. Support TOFlow in video frame interpolation

**New Features**

- Support AOT-GAN ([#677](https://github.com/open-mmlab/mmediting/pull/677))
- Use `--diff-seed` to set different torch seed on different rank ([#781](https://github.com/open-mmlab/mmediting/pull/781))
- Support streaming reading of frames in video interpolation demo ([#790](https://github.com/open-mmlab/mmediting/pull/790))
- Support dist_train without slurm ([#791](https://github.com/open-mmlab/mmediting/pull/791))
- Put LQ into CPU for restoration_video_demo ([#792](https://github.com/open-mmlab/mmediting/pull/792))
- Support gray normalization constant in EDSR ([#793](https://github.com/open-mmlab/mmediting/pull/793))
- Support TOFlow in video frame interpolation ([#806](https://github.com/open-mmlab/mmediting/pull/806), [#811](https://github.com/open-mmlab/mmediting/pull/811))
- Support seed in DistributedSampler and sync seed across ranks ([#815](https://github.com/open-mmlab/mmediting/pull/815))

**Bug Fixes**

- Update link in README files ([#782](https://github.com/open-mmlab/mmediting/pull/782), [#786](https://github.com/open-mmlab/mmediting/pull/786), [#819](https://github.com/open-mmlab/mmediting/pull/819), [#820](https://github.com/open-mmlab/mmediting/pull/820))
- Fix matting tutorial, and fix links to colab ([#795](https://github.com/open-mmlab/mmediting/pull/795))
- Invert `flip_ratio` in `RandomAffine` pipeline ([#799](https://github.com/open-mmlab/mmediting/pull/799))
- Update preprocess_div2k_dataset.py ([#801](https://github.com/open-mmlab/mmediting/pull/801))
- Update SR Colab Demo Installation Method and Set5 link ([#807](https://github.com/open-mmlab/mmediting/pull/807))
- Fix Y/GRB mistake in EDSR README ([#812](https://github.com/open-mmlab/mmediting/pull/812))
- Replace pytorch install command to conda in README(\_zh-CN).md ([#816](https://github.com/open-mmlab/mmediting/pull/816))

**Improvements**

- Update CI ([#650](https://github.com/open-mmlab/mmediting/pull/650))
- Update requirements.txt ([#725](https://github.com/open-mmlab/mmediting/pull/725), [#817](https://github.com/open-mmlab/mmediting/pull/817))
- Add Tutorial of dataset ([#758](https://github.com/open-mmlab/mmediting/pull/758)), pipeline ([#779](https://github.com/open-mmlab/mmediting/pull/779)), model ([#766](https://github.com/open-mmlab/mmediting/pull/758))
- Update index and TOC tree ([#767](https://github.com/open-mmlab/mmediting/pull/767))
- Make update_model_index.py compatible on windows ([#768](https://github.com/open-mmlab/mmediting/pull/768))
- Update doc build system ([#769](https://github.com/open-mmlab/mmediting/pull/769))
- Update keyword and classifier for setuptools ([#773](https://github.com/open-mmlab/mmediting/pull/773))
- Renovate installation ([#776](https://github.com/open-mmlab/mmediting/pull/776), [#800](https://github.com/open-mmlab/mmediting/pull/800))
- Update BasicVSR++ and RealBasicVSR docs ([#778](https://github.com/open-mmlab/mmediting/pull/778))
- Update citation ([#785](https://github.com/open-mmlab/mmediting/pull/785), [#787](https://github.com/open-mmlab/mmediting/pull/787))
- Regroup docs ([#788](https://github.com/open-mmlab/mmediting/pull/788))
- Use full name of config as 'Name' in metafile ([#798](https://github.com/open-mmlab/mmediting/pull/798))
- Update figure and video demo in README ([#802](https://github.com/open-mmlab/mmediting/pull/802))
- Add `clamp(0, 1)` in test of video frame interpolation ([#805](https://github.com/open-mmlab/mmediting/pull/805))
- Use hyphen for command line args in demo & tools ([#808](https://github.com/open-mmlab/mmediting/pull/808)), and keep underline for required arguments in python files ([#822](https://github.com/open-mmlab/mmediting/pull/822))
- Make dataset.pipeline a dedicated section in doc ([#813](https://github.com/open-mmlab/mmediting/pull/813))
- Update mmcv-full>=1.3.13 to support DCN on CPU ([#823](https://github.com/open-mmlab/mmediting/pull/823))

**Contributors**

@wangruohui @ckkelvinchan @Yshuo-Li @nijkah @wdmwhh @freepoet @quincylin1

## v0.13.0 (01/03/2022)

**Highlights**

1. Support CAIN
2. Support EDVR-L
3. Support running in Windows

**New Features**

- Add test-time ensemble for images and videos and support ensemble in BasicVSR series ([#585](https://github.com/open-mmlab/mmediting/pull/585))
- Support AOT-GAN (work in progress) ([#674](https://github.com/open-mmlab/mmediting/pull/674), [#675](https://github.com/open-mmlab/mmediting/pull/675), [#676](https://github.com/open-mmlab/mmediting/pull/676))
- Support CAIN ([#683](https://github.com/open-mmlab/mmediting/pull/683), [#691](https://github.com/open-mmlab/mmediting/pull/691), [#709](https://github.com/open-mmlab/mmediting/pull/709), [#713](https://github.com/open-mmlab/mmediting/pull/713))
- Add basic interpolater ([#687](https://github.com/open-mmlab/mmediting/pull/687))
- Add BaseVFIDataset and VFIVimeo90KDataset ([#695](https://github.com/open-mmlab/mmediting/pull/695), [#697](https://github.com/open-mmlab/mmediting/pull/697))
- Add video interpolation demo ([#688](https://github.com/open-mmlab/mmediting/pull/688), [#717](https://github.com/open-mmlab/mmediting/pull/717))
- Support various scales in RRDBNet ([#699](https://github.com/open-mmlab/mmediting/pull/699))
- Support Ref-SR inference ([#716](https://github.com/open-mmlab/mmediting/pull/716))
- Support EDVR-L on REDS ([#719](https://github.com/open-mmlab/mmediting/pull/719))
- Support CPU training ([#720](https://github.com/open-mmlab/mmediting/pull/720))
- Support running in Windows ([#732](https://github.com/open-mmlab/mmediting/pull/732), [#738](https://github.com/open-mmlab/mmediting/pull/738))
- Support DCN on CPU ([#735](https://github.com/open-mmlab/mmediting/pull/735))

**Bug Fixes**

- Fix link address in docs ([#703](https://github.com/open-mmlab/mmediting/pull/703), [#704](https://github.com/open-mmlab/mmediting/pull/704))
- Fix ARG `MMCV` in Dockerfile ([#708](https://github.com/open-mmlab/mmediting/pull/708))
- Fix file permission of non-executable files ([#718](https://github.com/open-mmlab/mmediting/pull/718))
- Fix some deprecation warning related to numpy ([#728](https://github.com/open-mmlab/mmediting/pull/728))
- Delete `__init__` in `TestVFIDataset` ([#731](https://github.com/open-mmlab/mmediting/pull/731))
- Fix data type in docstring of several Datasets ([#739](https://github.com/open-mmlab/mmediting/pull/739))
- Fix math notation in docstring ([#741](https://github.com/open-mmlab/mmediting/pull/741))
- Fix missing folders in copyright commit hook ([#754](https://github.com/open-mmlab/mmediting/pull/754))
- Delete duplicate test in loading ([#756](https://github.com/open-mmlab/mmediting/pull/756))

**Improvements**

- Update Pillow from 6.2.2 to 8.4 in CI ([#693](https://github.com/open-mmlab/mmediting/pull/693))
- Add argument 'repeat' to SRREDSMultipleGTDataset ([#672](https://github.com/open-mmlab/mmediting/pull/672))
- Deprecate the support for "python setup.py test" ([#701](https://github.com/open-mmlab/mmediting/pull/701))
- Add setup multi-processing both in train and test ([#707](https://github.com/open-mmlab/mmediting/pull/707))
- Add OpenMMLab website and platform links ([#710](https://github.com/open-mmlab/mmediting/pull/710))
- Refact README files of all methods ([#712](https://github.com/open-mmlab/mmediting/pull/712))
- Replace string version comparison with `package.version.parse` ([#723](https://github.com/open-mmlab/mmediting/pull/723))
- Add docs of Ref-SR demo and video frame interpolation demo ([#724](https://github.com/open-mmlab/mmediting/pull/724))
- Add interpolation and refact README.md ([#726](https://github.com/open-mmlab/mmediting/pull/726))
- Update isort version in pre-commit hook ([#727](https://github.com/open-mmlab/mmediting/pull/727))
- Redesign CI for Linux ([#734](https://github.com/open-mmlab/mmediting/pull/734))
- Update install.md ([#763](https://github.com/open-mmlab/mmediting/pull/763))
- Reorganizing OpenMMLab projects in readme ([#764](https://github.com/open-mmlab/mmediting/pull/764))
- Add deprecation message for deploy tools ([#765](https://github.com/open-mmlab/mmediting/pull/765))

**Contributors**

@wangruohui @ckkelvinchan @Yshuo-Li @quincylin1 @Juggernaut93 @anse3832 @nijkah

## v0.12.0 (31/12/2021)

**Highlights**

1. Support RealBasicVSR
2. Support Real-ESRGAN checkpoint

**New Features**

- Support video input and output in restoration demo ([#622](https://github.com/open-mmlab/mmediting/pull/622))
- Support RealBasicVSR ([#632](https://github.com/open-mmlab/mmediting/pull/632), [#633](https://github.com/open-mmlab/mmediting/pull/633), [#647](https://github.com/open-mmlab/mmediting/pull/647), [#680](https://github.com/open-mmlab/mmediting/pull/680))
- Support Real-ESRGAN checkpoint ([#635](https://github.com/open-mmlab/mmediting/pull/635))
- Support conversion to y-channel when loading images ([643](https://github.com/open-mmlab/mmediting/pull/643))
- Support random video compression during training ([#646](https://github.com/open-mmlab/mmediting/pull/646))
- Support crop sequence ([#648](https://github.com/open-mmlab/mmediting/pull/648))
- Support pixel_unshuffle ([#684](https://github.com/open-mmlab/mmediting/pull/684))

**Bug Fixes**

- Change 'target_size' for RandomResize from list to tuple ([#617](https://github.com/open-mmlab/mmediting/pull/617))
- Fix folder creation in preprocess_df2k_ost_dataset.py ([#623](https://github.com/open-mmlab/mmediting/pull/623))
- Change TDAN config path in README ([#625](https://github.com/open-mmlab/mmediting/pull/625))
- Change 'radius' to 'kernel_size' for UnsharpMasking in Real-ESRNet config ([#626](https://github.com/open-mmlab/mmediting/pull/626))
- Fix bug in MATLABLikeResize ([#630](https://github.com/open-mmlab/mmediting/pull/630))
- Fix 'flow_warp' comment ([#655](https://github.com/open-mmlab/mmediting/pull/655))
- Fix the error of Model Zoo and Datasets in docs ([#664](https://github.com/open-mmlab/mmediting/pull/664))
- Fix bug in 'random_degradations' ([#673](https://github.com/open-mmlab/mmediting/pull/673))
- Limit opencv-python version ([#689](https://github.com/open-mmlab/mmediting/pull/689))

**Improvements**

- Translate docs to Chinese ([#576](https://github.com/open-mmlab/mmediting/pull/576), [#577](https://github.com/open-mmlab/mmediting/pull/577), [#578](https://github.com/open-mmlab/mmediting/pull/578), [#579](https://github.com/open-mmlab/mmediting/pull/579), [#581](https://github.com/open-mmlab/mmediting/pull/581), [#582](https://github.com/open-mmlab/mmediting/pull/582), [#584](https://github.com/open-mmlab/mmediting/pull/584), [#587](https://github.com/open-mmlab/mmediting/pull/587), [#588](https://github.com/open-mmlab/mmediting/pull/588), [#589](https://github.com/open-mmlab/mmediting/pull/589), [#590](https://github.com/open-mmlab/mmediting/pull/590), [#591](https://github.com/open-mmlab/mmediting/pull/591), [#592](https://github.com/open-mmlab/mmediting/pull/592), [#593](https://github.com/open-mmlab/mmediting/pull/593), [#594](https://github.com/open-mmlab/mmediting/pull/594), [#595](https://github.com/open-mmlab/mmediting/pull/595), [#596](https://github.com/open-mmlab/mmediting/pull/596), [#641](https://github.com/open-mmlab/mmediting/pull/641), [#647](https://github.com/open-mmlab/mmediting/pull/647), [#656](https://github.com/open-mmlab/mmediting/pull/656), [#665](https://github.com/open-mmlab/mmediting/pull/665), [#666](https://github.com/open-mmlab/mmediting/pull/666))
- Add UNetDiscriminatorWithSpectralNorm ([#605](https://github.com/open-mmlab/mmediting/pull/605))
- Use PyTorch sphinx theme ([#607](https://github.com/open-mmlab/mmediting/pull/607), [#608](https://github.com/open-mmlab/mmediting/pull/608))
- Update mmcv ([#609](https://github.com/open-mmlab/mmediting/pull/609)), mmflow ([#621](https://github.com/open-mmlab/mmediting/pull/621)), mmfewshot ([#634](https://github.com/open-mmlab/mmediting/pull/634)) and mmhuman3d ([#649](https://github.com/open-mmlab/mmediting/pull/649)) in docs
- Convert minimum GCC version to 5.4 ([#612](https://github.com/open-mmlab/mmediting/pull/612))
- Add tiff in SRDataset IMG_EXTENSIONS ([#614](https://github.com/open-mmlab/mmediting/pull/614))
- Update metafile and update_model_index.py ([#615](https://github.com/open-mmlab/mmediting/pull/615))
- Update preprocess_df2k_ost_dataset.py ([#624](https://github.com/open-mmlab/mmediting/pull/624))
- Add Abstract to README ([#628](https://github.com/open-mmlab/mmediting/pull/628), [#636](https://github.com/open-mmlab/mmediting/pull/636))
- Align NIQE to MATLAB results ([#631](https://github.com/open-mmlab/mmediting/pull/631))
- Add official markdown lint hook ([#639](https://github.com/open-mmlab/mmediting/pull/639))
- Skip CI when some specific files were changed ([#640](https://github.com/open-mmlab/mmediting/pull/640))
- Update docs/conf.py ([#644](https://github.com/open-mmlab/mmediting/pull/644), [#651](https://github.com/open-mmlab/mmediting/pull/651))
- Try to create a symbolic link on windows ([#645](https://github.com/open-mmlab/mmediting/pull/645))
- Cancel previous runs that are not completed ([#650](https://github.com/open-mmlab/mmediting/pull/650))
- Update path of configs in demo.md and getting_started.md ([#658](https://github.com/open-mmlab/mmediting/pull/658), [#659](https://github.com/open-mmlab/mmediting/pull/659))
- Use mmcv root model registry ([#660](https://github.com/open-mmlab/mmediting/pull/660))
- Update README.md ([#654](https://github.com/open-mmlab/mmediting/pull/654), [#663](https://github.com/open-mmlab/mmediting/pull/663))
- Refactor the structure of documentation ([#668](https://github.com/open-mmlab/mmediting/pull/668))
- Add script to crop REDS images into sub-images for faster IO ([#669](https://github.com/open-mmlab/mmediting/pull/669))
- Capitalize the first letter of the task name in the metafile ([#678](https://github.com/open-mmlab/mmediting/pull/678))
- Update FixedCrop for cropping image sequence ([#682](https://github.com/open-mmlab/mmediting/pull/682))

## v0.11.0 (03/11/2021)

**Highlights**

- GLEAN for blind face image restoration #530
- Real-ESRGAN model #546

**New Features**

- Exponential Moving Average Hook #542
- Support DF2K_OST dataset #566

**Improvements**

- Add MATLAB-like bicubic interpolation #507
- Support random degradations during training #504
- Support torchserve #568

## v0.10.0 (12/08/2021).

**Highlights**

1. Support LIIF-RDN (CVPR'2021)
2. Support BasicVSR++ (NTIRE'2021)

**New Features**

- Support loading annotation from file for video SR datasets ([#423](https://github.com/open-mmlab/mmediting/pull/423))
- Support persistent worker ([#426](https://github.com/open-mmlab/mmediting/pull/426))
- Support LIIF-RDN ([#428](https://github.com/open-mmlab/mmediting/pull/428), [#440](https://github.com/open-mmlab/mmediting/pull/440))
- Support BasicVSR++ ([#451](https://github.com/open-mmlab/mmediting/pull/451), [#467](https://github.com/open-mmlab/mmediting/pull/467))
- Support mim ([#455](https://github.com/open-mmlab/mmediting/pull/455))

**Bug Fixes**

- Fix bug in stat.py ([#420](https://github.com/open-mmlab/mmediting/pull/420))
- Fix astype error in function tensor2img ([#429](https://github.com/open-mmlab/mmediting/pull/429))
- Fix device error caused by torch.new_tensor when pytorch >= 1.7 ([#465](https://github.com/open-mmlab/mmediting/pull/465))
- Fix \_non_dist_train in .mmedit/apis/train.py ([#473](https://github.com/open-mmlab/mmediting/pull/473))
- Fix multi-node distributed test ([#478](https://github.com/open-mmlab/mmediting/pull/478))

**Breaking Changes**

- Refactor LIIF for pytorch2onnx ([#425](https://github.com/open-mmlab/mmediting/pull/425))

**Improvements**

- Update Chinese docs ([#415](https://github.com/open-mmlab/mmediting/pull/415), [#416](https://github.com/open-mmlab/mmediting/pull/416), [#418](https://github.com/open-mmlab/mmediting/pull/418), [#421](https://github.com/open-mmlab/mmediting/pull/421), [#424](https://github.com/open-mmlab/mmediting/pull/424), [#431](https://github.com/open-mmlab/mmediting/pull/431), [#442](https://github.com/open-mmlab/mmediting/pull/442))
- Add CI of pytorch 1.9.0 ([#444](https://github.com/open-mmlab/mmediting/pull/444))
- Refactor README.md of configs ([#452](https://github.com/open-mmlab/mmediting/pull/452))
- Avoid loading pretrained VGG in unittest ([#466](https://github.com/open-mmlab/mmediting/pull/466))
- Support specifying scales in preprocessing div2k dataset ([#472](https://github.com/open-mmlab/mmediting/pull/472))
- Support all formats in readthedocs ([#479](https://github.com/open-mmlab/mmediting/pull/479))
- Use version_info of mmcv ([#480](https://github.com/open-mmlab/mmediting/pull/480))
- Remove unnecessary codes in restoration_video_demo.py ([#484](https://github.com/open-mmlab/mmediting/pull/484))
- Change priority of DistEvalIterHook to 'LOW' ([#489](https://github.com/open-mmlab/mmediting/pull/489))
- Reset resource limit ([#491](https://github.com/open-mmlab/mmediting/pull/491))
- Update QQ QR code in README_CN.md ([#494](https://github.com/open-mmlab/mmediting/pull/494))
- Add `myst_parser` ([#495](https://github.com/open-mmlab/mmediting/pull/495))
- Add license header ([#496](https://github.com/open-mmlab/mmediting/pull/496))
- Fix typo of StyleGAN modules ([#427](https://github.com/open-mmlab/mmediting/pull/427))
- Fix typo in docs/demo.md ([#453](https://github.com/open-mmlab/mmediting/pull/453), [#454](https://github.com/open-mmlab/mmediting/pull/454))
- Fix typo in tools/data/super-resolution/reds/README.md ([#469](https://github.com/open-mmlab/mmediting/pull/469))

## v0.9.0 (30/06/2021).

**Highlights**

1. Support DIC and DIC-GAN (CVPR'2020)
2. Support GLEAN Cat 8x (CVPR'2021)
3. Support TTSR-GAN (CVPR'2020)
4. Add colab tutorial for super-resolution

**New Features**

- Add DIC ([#342](https://github.com/open-mmlab/mmediting/pull/342), [#345](https://github.com/open-mmlab/mmediting/pull/345), [#348](https://github.com/open-mmlab/mmediting/pull/348), [#350](https://github.com/open-mmlab/mmediting/pull/350), [#351](https://github.com/open-mmlab/mmediting/pull/351), [#357](https://github.com/open-mmlab/mmediting/pull/357), [#363](https://github.com/open-mmlab/mmediting/pull/363), [#365](https://github.com/open-mmlab/mmediting/pull/365), [#366](https://github.com/open-mmlab/mmediting/pull/366))
- Add SRFolderMultipleGTDataset ([#355](https://github.com/open-mmlab/mmediting/pull/355))
- Add GLEAN Cat 8x ([#367](https://github.com/open-mmlab/mmediting/pull/367))
- Add SRFolderVideoDataset ([#370](https://github.com/open-mmlab/mmediting/pull/370))
- Add colab tutorial for super-resolution ([#380](https://github.com/open-mmlab/mmediting/pull/380))
- Add TTSR-GAN ([#372](https://github.com/open-mmlab/mmediting/pull/372), [#381](https://github.com/open-mmlab/mmediting/pull/381), [#383](https://github.com/open-mmlab/mmediting/pull/383), [#398](https://github.com/open-mmlab/mmediting/pull/398))
- Add DIC-GAN ([#392](https://github.com/open-mmlab/mmediting/pull/392), [#393](https://github.com/open-mmlab/mmediting/pull/393), [#394](https://github.com/open-mmlab/mmediting/pull/394))

**Bug Fixes**

- Fix bug in restoration_video_inference.py ([#379](https://github.com/open-mmlab/mmediting/pull/379))
- Fix Config of LIIF ([#368](https://github.com/open-mmlab/mmediting/pull/368))
- Change the path to pre-trained EDVR-M ([#396](https://github.com/open-mmlab/mmediting/pull/396))
- Fix normalization in restoration_video_inference ([#406](https://github.com/open-mmlab/mmediting/pull/406))
- Fix \[brush_stroke_mask\] error in unittest ([#409](https://github.com/open-mmlab/mmediting/pull/409))

**Breaking Changes**

- Change mmcv minimum version to v1.3 ([#378](https://github.com/open-mmlab/mmediting/pull/378))

**Improvements**

- Correct Typos in code ([#371](https://github.com/open-mmlab/mmediting/pull/371))
- Add Custom_hooks ([#362](https://github.com/open-mmlab/mmediting/pull/362))
- Refactor unittest folder structure ([#386](https://github.com/open-mmlab/mmediting/pull/386))
- Add documents and download link for Vid4 ([#399](https://github.com/open-mmlab/mmediting/pull/399))
- Update model zoo for documents ([#400](https://github.com/open-mmlab/mmediting/pull/400))
- Update metafile ([407](https://github.com/open-mmlab/mmediting/pull/407))

## v0.8.0 (31/05/2021).

**Highlights**

1. Support GLEAN (CVPR'2021)
2. Support TTSR (CVPR'2020)
3. Support TDAN (CVPR'2020)

**New Features**

- Add GLEAN ([#296](https://github.com/open-mmlab/mmediting/pull/296), [#332](https://github.com/open-mmlab/mmediting/pull/332))
- Support PWD metafile ([#298](https://github.com/open-mmlab/mmediting/pull/298))
- Support CropLike in pipeline ([#299](https://github.com/open-mmlab/mmediting/pull/299))
- Add TTSR ([#301](https://github.com/open-mmlab/mmediting/pull/301), [#304](https://github.com/open-mmlab/mmediting/pull/304), [#307](https://github.com/open-mmlab/mmediting/pull/307), [#311](https://github.com/open-mmlab/mmediting/pull/311), [#311](https://github.com/open-mmlab/mmediting/pull/311), [#312](https://github.com/open-mmlab/mmediting/pull/312), [#313](https://github.com/open-mmlab/mmediting/pull/313), [#314](https://github.com/open-mmlab/mmediting/pull/314), [#321](https://github.com/open-mmlab/mmediting/pull/321), [#326](https://github.com/open-mmlab/mmediting/pull/326), [#327](https://github.com/open-mmlab/mmediting/pull/327))
- Add TDAN ([#316](https://github.com/open-mmlab/mmediting/pull/316), [#334](https://github.com/open-mmlab/mmediting/pull/334), [#347](https://github.com/open-mmlab/mmediting/pull/347))
- Add onnx2tensorrt ([#317](https://github.com/open-mmlab/mmediting/pull/317))
- Add tensorrt evaluation ([#328](https://github.com/open-mmlab/mmediting/pull/328))
- Add SRFacicalLandmarkDataset ([#329](https://github.com/open-mmlab/mmediting/pull/329))
- Add key point auxiliary model for DIC ([#336](https://github.com/open-mmlab/mmediting/pull/336), [#341](https://github.com/open-mmlab/mmediting/pull/341))
- Add demo for video super-resolution methods ([#275](https://github.com/open-mmlab/mmediting/pull/275))
- Add SR Folder Ref Dataset ([#292](https://github.com/open-mmlab/mmediting/pull/292))
- Support FLOPs calculation of video SR models ([#309](https://github.com/open-mmlab/mmediting/pull/309))

**Bug Fixes**

- Fix find_unused_parameters in PyTorch 1.8 for BasicVSR ([#290](https://github.com/open-mmlab/mmediting/pull/290))
- Fix error in publish_model.py for pt>=1.6 ([#291](https://github.com/open-mmlab/mmediting/pull/291))
- Fix PSNR when input is uint8 ([#294](https://github.com/open-mmlab/mmediting/pull/294))

**Improvements**

- Support backend in LoadImageFromFile ([#293](https://github.com/open-mmlab/mmediting/pull/293), [#303](https://github.com/open-mmlab/mmediting/pull/303))
- Update `metric_average_mode` of video SR dataset ([#319](https://github.com/open-mmlab/mmediting/pull/319))
- Add error message in restoration_demo.py ([324](https://github.com/open-mmlab/mmediting/pull/324))
- Minor correction in getting_started.md ([#339](https://github.com/open-mmlab/mmediting/pull/339))
- Update description for Vimeo90K ([#349](https://github.com/open-mmlab/mmediting/pull/349))
- Support start_index in GenerateSegmentIndices ([#338](https://github.com/open-mmlab/mmediting/pull/338))
- Support different filename templates in GenerateSegmentIndices ([#325](https://github.com/open-mmlab/mmediting/pull/325))
- Support resize by scale-factor ([#295](https://github.com/open-mmlab/mmediting/pull/295), [#310](https://github.com/open-mmlab/mmediting/pull/310))

## v0.7.0 (30/04/2021).

**Highlights**

1. Support BasicVSR (CVPR'2021)
2. Support IconVSR (CVPR'2021)
3. Support RDN (CVPR'2018)
4. Add onnx evaluation tool

**New Features**

- Add RDN ([#233](https://github.com/open-mmlab/mmediting/pull/233), [#260](https://github.com/open-mmlab/mmediting/pull/260), [#280](https://github.com/open-mmlab/mmediting/pull/280))
- Add MultipleGT datasets ([#238](https://github.com/open-mmlab/mmediting/pull/238))
- Add BasicVSR and IconVSR ([#245](https://github.com/open-mmlab/mmediting/pull/245), [#252](https://github.com/open-mmlab/mmediting/pull/252), [#253](https://github.com/open-mmlab/mmediting/pull/253), [#254](https://github.com/open-mmlab/mmediting/pull/254), [#264](https://github.com/open-mmlab/mmediting/pull/264), [#274](https://github.com/open-mmlab/mmediting/pull/274), [#258](https://github.com/open-mmlab/mmediting/pull/258), [#252](https://github.com/open-mmlab/mmediting/pull/252), [#264](https://github.com/open-mmlab/mmediting/pull/264))
- Add onnx evaluation tool ([#279](https://github.com/open-mmlab/mmediting/pull/279))

**Bug Fixes**

- Fix onnx conversion of maxunpool2d ([#243](https://github.com/open-mmlab/mmediting/pull/243))
- Fix inpainting in `demo.md` ([#248](https://github.com/open-mmlab/mmediting/pull/248))
- Tiny fix of config file of EDSR ([#251](https://github.com/open-mmlab/mmediting/pull/251))
- Fix link in README ([#256](https://github.com/open-mmlab/mmediting/pull/256))
- Fix restoration_inference key missing bug ([#270](https://github.com/open-mmlab/mmediting/pull/270))
- Fix the usage of channel_order in `loading.py` ([#271](https://github.com/open-mmlab/mmediting/pull/271))
- Fix the command of inpainting ([#278](https://github.com/open-mmlab/mmediting/pull/278))
- Fix `preprocess_vimeo90k_dataset.py` args name ([#281](https://github.com/open-mmlab/mmediting/pull/281))

**Improvements**

- Support `empty_cache` option in `test.py` ([#261](https://github.com/open-mmlab/mmediting/pull/261))
- Update projects in README ([#249](https://github.com/open-mmlab/mmediting/pull/249), [#276](https://github.com/open-mmlab/mmediting/pull/276))
- Support Y-channel PSNR and SSIM ([#250](https://github.com/open-mmlab/mmediting/pull/250))
- Add zh-CN README ([#262](https://github.com/open-mmlab/mmediting/pull/262))
- Update pytorch2onnx doc ([#265](https://github.com/open-mmlab/mmediting/pull/265))
- Remove extra quotation in English readme ([#268](https://github.com/open-mmlab/mmediting/pull/268))
- Change tags to comment ([#269](https://github.com/open-mmlab/mmediting/pull/269))
- List `model zoo` in README ([#284](https://github.com/open-mmlab/mmediting/pull/284), [#285](https://github.com/open-mmlab/mmediting/pull/285), [#286](https://github.com/open-mmlab/mmediting/pull/286))

## v0.6.0 (08/04/2021).

**Highlights**

1. Support Local Implicit Image Function (LIIF)
2. Support exporting DIM and GCA from Pytorch to ONNX

**New Features**

- Add readthedocs config files and fix docstring ([#92](https://github.com/open-mmlab/mmediting/pull/92))
- Add github action file ([#94](https://github.com/open-mmlab/mmediting/pull/94))
- Support exporting DIM and GCA from Pytorch to ONNX ([#105](https://github.com/open-mmlab/mmediting/pull/105))
- Support concatenating datasets ([#106](https://github.com/open-mmlab/mmediting/pull/106))
- Support `non_dist_train` validation ([#110](https://github.com/open-mmlab/mmediting/pull/110))
- Add matting colab tutorial ([#111](https://github.com/open-mmlab/mmediting/pull/111))
- Support niqe metric ([#114](https://github.com/open-mmlab/mmediting/pull/114))
- Support PoolDataLoader for parrots ([#134](https://github.com/open-mmlab/mmediting/pull/134))
- Support collect-env ([#137](https://github.com/open-mmlab/mmediting/pull/137), [#143](https://github.com/open-mmlab/mmediting/pull/143))
- Support pt1.6 cpu/gpu in CI ([#138](https://github.com/open-mmlab/mmediting/pull/138))
- Support fp16 ([139](https://github.com/open-mmlab/mmediting/pull/139), [#144](https://github.com/open-mmlab/mmediting/pull/144))
- Support publishing to pypi ([#149](https://github.com/open-mmlab/mmediting/pull/149))
- Add modelzoo statistics ([#171](https://github.com/open-mmlab/mmediting/pull/171), [#182](https://github.com/open-mmlab/mmediting/pull/182), [#186](https://github.com/open-mmlab/mmediting/pull/186))
- Add doc of datasets ([194](https://github.com/open-mmlab/mmediting/pull/194))
- Support extended foreground option. ([#195](https://github.com/open-mmlab/mmediting/pull/195), [#199](https://github.com/open-mmlab/mmediting/pull/199), [#200](https://github.com/open-mmlab/mmediting/pull/200), [#210](https://github.com/open-mmlab/mmediting/pull/210))
- Support nn.MaxUnpool2d ([#196](https://github.com/open-mmlab/mmediting/pull/196))
- Add some FBA components ([#203](https://github.com/open-mmlab/mmediting/pull/203), [#209](https://github.com/open-mmlab/mmediting/pull/209), [#215](https://github.com/open-mmlab/mmediting/pull/215), [#220](https://github.com/open-mmlab/mmediting/pull/220))
- Support random down sampling in pipeline ([#222](https://github.com/open-mmlab/mmediting/pull/222))
- Support SR folder GT Dataset ([#223](https://github.com/open-mmlab/mmediting/pull/223))
- Support Local Implicit Image Function (LIIF) ([#224](https://github.com/open-mmlab/mmediting/pull/224), [#226](https://github.com/open-mmlab/mmediting/pull/226), [#227](https://github.com/open-mmlab/mmediting/pull/227), [#234](https://github.com/open-mmlab/mmediting/pull/234), [#239](https://github.com/open-mmlab/mmediting/pull/239))

**Bug Fixes**

- Fix `_non_dist_train` in train api ([#104](https://github.com/open-mmlab/mmediting/pull/104))
- Fix setup and CI ([#109](https://github.com/open-mmlab/mmediting/pull/109))
- Fix redundant loop bug in Normalize ([#121](https://github.com/open-mmlab/mmediting/pull/121))
- Fix `get_hash` in `setup.py` ([#124](https://github.com/open-mmlab/mmediting/pull/124))
- Fix `tool/preprocess_reds_dataset.py` ([#148](https://github.com/open-mmlab/mmediting/pull/148))
- Fix slurm train tutorial in `getting_started.md` ([#162](https://github.com/open-mmlab/mmediting/pull/162))
- Fix pip install bug ([#173](https://github.com/open-mmlab/mmediting/pull/173))
- Fix bug in config file ([#185](https://github.com/open-mmlab/mmediting/pull/185))
- Fix broken links of datasets ([#236](https://github.com/open-mmlab/mmediting/pull/236))
- Fix broken links of model zoo ([#242](https://github.com/open-mmlab/mmediting/pull/242))

**Breaking Changes**

- Refactor data loader configs ([#201](https://github.com/open-mmlab/mmediting/pull/201))

**Improvements**

- Updata requirements.txt ([#95](https://github.com/open-mmlab/mmediting/pull/95), [#100](https://github.com/open-mmlab/mmediting/pull/100))
- Update teaser ([#96](https://github.com/open-mmlab/mmediting/pull/96))
- Updata README ([#93](https://github.com/open-mmlab/mmediting/pull/93), [#97](https://github.com/open-mmlab/mmediting/pull/97), [#98](https://github.com/open-mmlab/mmediting/pull/98), [#152](https://github.com/open-mmlab/mmediting/pull/152))
- Updata model_zoo ([#101](https://github.com/open-mmlab/mmediting/pull/101))
- Fix typos ([#102](https://github.com/open-mmlab/mmediting/pull/102), [#188](https://github.com/open-mmlab/mmediting/pull/188), [#191](https://github.com/open-mmlab/mmediting/pull/191), [#197](https://github.com/open-mmlab/mmediting/pull/197), [#208](https://github.com/open-mmlab/mmediting/pull/208))
- Adopt adjust_gamma from skimage and reduce dependencies ([#112](https://github.com/open-mmlab/mmediting/pull/112))
- remove `.gitlab-ci.yml` ([#113](https://github.com/open-mmlab/mmediting/pull/113))
- Update import of first party ([#115](https://github.com/open-mmlab/mmediting/pull/115))
- Remove citation and contact ([#122](https://github.com/open-mmlab/mmediting/pull/122))
- Update version file ([#136](https://github.com/open-mmlab/mmediting/pull/136))
- Update download url ([#141](https://github.com/open-mmlab/mmediting/pull/141))
- Update `setup.py` ([#150](https://github.com/open-mmlab/mmediting/pull/150))
- Update the highest version of supported mmcv ([#153](https://github.com/open-mmlab/mmediting/pull/153), [#154](https://github.com/open-mmlab/mmediting/pull/154))
- modify `Crop` to handle a sequence of video frames ([#164](https://github.com/open-mmlab/mmediting/pull/164))
- Add links to other mm projects ([#179](https://github.com/open-mmlab/mmediting/pull/179), [#180](https://github.com/open-mmlab/mmediting/pull/180))
- Add config type ([#181](https://github.com/open-mmlab/mmediting/pull/181))
- Refactor docs ([#184](https://github.com/open-mmlab/mmediting/pull/184))
- Add config link ([#187](https://github.com/open-mmlab/mmediting/pull/187))
- Update file structure ([#192](https://github.com/open-mmlab/mmediting/pull/192))
- Update config doc ([#202](https://github.com/open-mmlab/mmediting/pull/202))
- Update `slurm_train.md` script ([#204](https://github.com/open-mmlab/mmediting/pull/204))
- Improve code style ([#206](https://github.com/open-mmlab/mmediting/pull/206), [#207](https://github.com/open-mmlab/mmediting/pull/207))
- Use `file_client` in CompositeFg ([#212](https://github.com/open-mmlab/mmediting/pull/212))
- Replace `random` with `numpy.random` ([#213](https://github.com/open-mmlab/mmediting/pull/213))
- Refactor `loader_cfg` ([#214](https://github.com/open-mmlab/mmediting/pull/214))

## v0.5.0 (09/07/2020).

Note that **MMSR** has been merged into this repo, as a part of MMEditing.
With elaborate designs of the new framework and careful implementations,
hope MMEditing could provide better experience.
