# Changelog

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
- Fix _non_dist_train in .mmedit/apis/train.py ([#473](https://github.com/open-mmlab/mmediting/pull/473))
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
- Fix [brush_stroke_mask] error in unittest ([#409](https://github.com/open-mmlab/mmediting/pull/409))

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
