# Example Project

This is an example README for community `projects/`. You can write your README in your own project. Here are
some recommended parts of a README for others to understand and use your project, you can copy or modify them
according to your project.

## Description \[required\]

You can share any information you would like others to know. For example:

```
Author: @xxx.

This is an implementation of \[XXX\].
```

## Usage \[required\]

### Setup Environment \[required\]

Please refer to [Get Started](https://mmagic.readthedocs.io/en/latest/get_started/I.html) to install
MMagic.

At first, add the current folder to `PYTHONPATH`, so that Python can find your code. Run command in the current directory to add it.

> Please run it every time after you opened a new shell.

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Data Preparation \[optional\]

Prepare the ImageNet-2012 dataset according to the [instruction](https://mmagic.readthedocs.io/en/latest/user_guides/dataset_prepare.html#imagenet).

### Training commands \[optional\]

**To train with single GPU:**

```bash
mim train mmagic configs/examplenet_8xb32_in1k.py
```

**To train with multiple GPUs:**

```bash
mim train mmagic configs/examplenet_8xb32_in1k.py --launcher pytorch --gpus 8
```

**To train with multiple GPUs by slurm:**

```bash
mim train mmagic configs/examplenet_8xb32_in1k.py --launcher slurm \
    --gpus 16 --gpus-per-node 8 --partition $PARTITION
```

### Testing commands \[required\]

**To test with single GPU:**

```bash
mim test mmagic configs/examplenet_8xb32_in1k.py $CHECKPOINT
```

**To test with multiple GPUs:**

```bash
mim test mmagic configs/examplenet_8xb32_in1k.py $CHECKPOINT --launcher pytorch --gpus 8
```

**To test with multiple GPUs by slurm:**

```bash
mim test mmagic configs/examplenet_8xb32_in1k.py $CHECKPOINT --launcher slurm \
    --gpus 16 --gpus-per-node 8 --partition $PARTITION
```

## Results \[required\]

|       Model        |   Pretrain   | Top-1 (%) | Top-5 (%) |                   Config                   |                Download                |
| :----------------: | :----------: | :-------: | :-------: | :----------------------------------------: | :------------------------------------: |
|  ExampleNet-tiny   | From scratch |   82.33   |   96.15   | [config](configs/examplenet_8xb32_in1k.py) | [model](MODEL-LINK) \| [log](LOG-LINK) |
| ExampleNet-small\* | From scratch |   83.63   |   96.51   | [config](configs/examplenet_8xb32_in1k.py) |          [model](MODEL-LINK)           |
| ExampleNet-base\*  | From scratch |   84.34   |   96.86   | [config](configs/examplenet_8xb32_in1k.py) |          [model](MODEL-LINK)           |

*Models with * are converted from the [official repo](REPO-LINK). The config files of these models are only for inference. We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results.*

You can also paste some visual results here if the model doesn't have quantitative results.

## Citation \[required\]

<!-- Replace to the citation of the paper your project refers to. -->

```bibtex
@misc{mmagic2023,
    title = {{MMagic}: {OpenMMLab} Multimodal Advanced, Generative, and Intelligent Creation Toolbox},
    author = {{MMagic Contributors}},
    howpublished = {\url{https://github.com/open-mmlab/mmagic}},
    year = {2023}
}
```

## Checklist \[required\]

Here is a checklist of this project's progress. And you can ignore this part if you don't plan to contribute
to MMagic projects.

- [ ] Milestone 1: PR-ready, and acceptable to be one of the `projects/`.

  - [ ] Finish the code

    <!-- The code's design shall follow existing interfaces and convention. For example, each model component should be registered into `mmagic.registry.MODELS` and configurable via a config file. -->

  - [ ] Basic docstrings & proper citation

    <!-- Each major class should contains a docstring, describing its functionality and arguments. If your code is copied or modified from other open-source projects, don't forget to cite the source project in docstring and make sure your behavior is not against its license. Typically, we do not accept any code snippet under GPL license. [A Short Guide to Open Source Licenses](https://medium.com/nationwide-technology/a-short-guide-to-open-source-licenses-cf5b1c329edd) -->

  - [ ] Converted checkpoint and results (Only for reproduction)

    <!-- If you are reproducing the result from a paper, make sure the model in the project can match that results. Also please provide checkpoint links or a checkpoint conversion script for others to get the pre-trained model. -->

- [ ] Milestone 2: Indicates a successful model implementation.

  - [ ] Training results

    <!-- If you are reproducing the result from a paper, train your model from scratch and verified that the final result can match the original result. -->

- [ ] Milestone 3: Good to be a part of our core package!

  - [ ] Unit tests

    <!-- Unit tests for the major module are required. [Example](https://github.com/open-mmlab/mmagic/blob/main/tests/test_models/test_backbones/test_vision_transformer.py) -->

  - [ ] Code style

    <!-- Refactor your code according to reviewer's comment. -->

  - [ ] `metafile.yml` and `README.md`

    <!-- It will used for mmagic to acquire your models. [Example](https://github.com/open-mmlab/mmagic/blob/main/configs/mvit/metafile.yml). In particular, you may have to refactor this README into a standard one. [Example](https://github.com/open-mmlab/mmagic/blob/main/configs/swin_transformer/README.md) -->
