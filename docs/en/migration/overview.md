# Overview

This section introduce the following contents in terms of migration from MMEditing 0.x

- [New dependencies](#new-dependencies)
- [Overall structures](#overall-structures)

## New dependencies

MMEdit main depends on some new packages, you can prepare a new clean environment and install again according to the [install tutorial](../get_started/install.md). Or install the below packages manually.

1. [MMEngine](https://github.com/open-mmlab/mmengine): MMEngine is the core the OpenMMLab 2.0 architecture, and we splited many compentents unrelated to computer vision from MMCV to MMEngine.
2. [MMCV](https://github.com/open-mmlab/mmcv/tree/dev-2.x): The computer vision package of OpenMMLab. This is not a new dependency, but you need to upgrade it to above 2.0.0rc0 version.
3. [rich](https://github.com/Textualize/rich): A terminal formatting package, and we use it to beautify some outputs in the terminal.

## Overall structures

We refactor overall structures in MMEdit main as following.

- The  `core` in the old versions of MMEdit is split into `engine`, `evaluation`, `structures`, and `visualization`
- The `pipelines` of `datasets` in the old versions of MMEdit is refactored to `transforms`
- The `models` in MMedit main is refactored to five parts: `base_models`, `data_preprocessors`, `editors`, `layers` and `losses`.

## Other config settings

We rename config file to new template: `{model_settings}_{module_setting}_{training_setting}_{datasets_info}`.

More details of config are shown in [config guides](../user_guides/config.md).
