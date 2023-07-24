# Overview

This section introduces the following contents in terms of migration from MMEditing 0.x

- [Overview](#overview)
  - [New dependencies](#new-dependencies)
  - [Overall structures](#overall-structures)
  - [Other config settings](#other-config-settings)

## New dependencies

MMagic 1.x depends on some new packages, you can prepare a new clean environment and install it again according to the [install tutorial](../get_started/install.md).

## Overall structures

We refactor overall structures in MMagic 1.x as follows.

- The  `core` in the old versions of MMEdit is split into `engine`, `evaluation`, `structures`, and `visualization`
- The `pipelines` of `datasets` in the old versions of MMEdit is refactored to `transforms`
- The `models` in MMagic 1.x is refactored to six parts: `archs`, `base_models`, `data_preprocessors`, `editors`, `diffusion_schedulers`, and `losses`.

## Other config settings

We rename the config file to the new template: `{model_settings}_{module_setting}_{training_setting}_{datasets_info}`.

More details of config are shown in [config guides](../user_guides/config.md).
