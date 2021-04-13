# Changelog

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
- Refector docs ([#184](https://github.com/open-mmlab/mmediting/pull/184))
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
