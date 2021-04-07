# Changelog

## v0.6.0 (07/04/2021).

**Highlights**

1.

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
- Add modelzoo statistics ([#171](https://github.com/open-mmlab/mmediting/pull/171))

**Bug Fixes**
- Fix `_non_dist_train` in train api ([#104](https://github.com/open-mmlab/mmediting/pull/104))
- Fix setup and CI ([#109](https://github.com/open-mmlab/mmediting/pull/109))
- Fix redundant loop bug in Normalize ([#121](https://github.com/open-mmlab/mmediting/pull/121))
- Fix `get_hash` in `setup.py` ([#124](https://github.com/open-mmlab/mmediting/pull/124))
- Fix `tool/preprocess_reds_dataset.py` ([#148](https://github.com/open-mmlab/mmediting/pull/148))
- fix pip install bug ([#173](https://github.com/open-mmlab/mmediting/pull/173))

**Breaking Changes**



**Improvements**

- Updata requirements.txt ([#95](https://github.com/open-mmlab/mmediting/pull/95), [#100](https://github.com/open-mmlab/mmediting/pull/100))
- Update teaser ([#96](https://github.com/open-mmlab/mmediting/pull/96))
- Updata README ([#93](https://github.com/open-mmlab/mmediting/pull/93), [#97](https://github.com/open-mmlab/mmediting/pull/97), [#98](https://github.com/open-mmlab/mmediting/pull/98), [#152](https://github.com/open-mmlab/mmediting/pull/152))
- Updata model_zoo ([#101](https://github.com/open-mmlab/mmediting/pull/101))
- Fix typos ([#102](https://github.com/open-mmlab/mmediting/pull/102))
- Adopt adjust_gamma from skimage and reduce dependencies ([#112](https://github.com/open-mmlab/mmediting/pull/112))
- remove `.gitlab-ci.yml` ([#113](https://github.com/open-mmlab/mmediting/pull/113))
- Update import of first party ([#115](https://github.com/open-mmlab/mmediting/pull/115))
- Remove citation and contact ([#122](https://github.com/open-mmlab/mmediting/pull/122))
- Update version file ([#136](https://github.com/open-mmlab/mmediting/pull/136))
- Update download url ([#141](https://github.com/open-mmlab/mmediting/pull/141))
- Update `setup.py` ([#150](https://github.com/open-mmlab/mmediting/pull/150))
- Update the highest version of supported mmcv ([#153](https://github.com/open-mmlab/mmediting/pull/153), [#154](https://github.com/open-mmlab/mmediting/pull/154))
- modify `Crop` to handle a sequence of video frames ([#164](https://github.com/open-mmlab/mmediting/pull/164))
- Add links to other mm projects ([#179](https://github.com/open-mmlab/mmediting/pull/179))

## v0.5.0 (09/07/2020).

Note that **MMSR** has been merged into this repo, as a part of MMEditing.
With elaborate designs of the new framework and careful implementations,
hope MMEditing could provide better experience.
