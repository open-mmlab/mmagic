# Style-Based Global Appearance Flow for Virtual Try-On (CVPR 2022)

## Description

Awesome try-on desplays are like this:

![image1](examples/000010_0.png)

```
Author: @FerryHuang.

This is an implementation of https://github.com/SenHe/Flow-Style-VTON adapting to mmediting. Only inference is supported so far.
```

## Usage

### Setup Environment

Please refer to [Get Started](https://mmediting.readthedocs.io/en/latest/get_started/I.html) to install
MMEditing.

At first, add the current folder to `PYTHONPATH`, so that Python can find your code. Run command in the current directory to add it.

> Please run it every time after you opened a new shell.

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Data Preparation

Please check the [official repo](https://github.com/SenHe/Flow-Style-VTON) and download test-set and pretrained checkpoints and put them under the folder projects/flow_style_vton

### Testing commands

**To test with single GPU:**

```bash
cd projects/flow_style_vton
python inference.py
```

Expectedly, two folders will be made im_gar_flow_wg and our_t_results, containing the
try-on procedures and the final results, respectively.

## Citation

<!-- Replace to the citation of the paper your project refers to. -->

```bibtex
@inproceedings{he2022fs_vton,
  title={Style-Based Global Appearance Flow for Virtual Try-On},
  author={He, Sen and Song, Yi-Zhe and Xiang, Tao},
  booktitle={CVPR},
  year={2022}
}
```

## Checklist \[required\]

Here is a checklist of this project's progress. And you can ignore this part if you don't plan to contribute
to MMediting projects.

- [ ] Milestone 1: PR-ready, and acceptable to be one of the `projects/`.

  - [ ] Finish the code

    <!-- The code's design shall follow existing interfaces and convention. For example, each model component should be registered into `mmedit.registry.MODELS` and configurable via a config file. -->

  - [ ] Basic docstrings & proper citation

    <!-- Each major class should contains a docstring, describing its functionality and arguments. If your code is copied or modified from other open-source projects, don't forget to cite the source project in docstring and make sure your behavior is not against its license. Typically, we do not accept any code snippet under GPL license. [A Short Guide to Open Source Licenses](https://medium.com/nationwide-technology/a-short-guide-to-open-source-licenses-cf5b1c329edd) -->

  - [ ] Converted checkpoint and results (Only for reproduction)

    <!-- If you are reproducing the result from a paper, make sure the model in the project can match that results. Also please provide checkpoint links or a checkpoint conversion script for others to get the pre-trained model. -->

- [ ] Milestone 2: Indicates a successful model implementation.

  - [ ] Training results

    <!-- If you are reproducing the result from a paper, train your model from scratch and verified that the final result can match the original result. -->

- [ ] Milestone 3: Good to be a part of our core package!

  - [ ] Unit tests

    <!-- Unit tests for the major module are required. [Example](https://github.com/open-mmlab/mmediting/blob/main/tests/test_models/test_backbones/test_vision_transformer.py) -->

  - [ ] Code style

    <!-- Refactor your code according to reviewer's comment. -->

  - [ ] `metafile.yml` and `README.md`

    <!-- It will used for mmediting to acquire your models. [Example](https://github.com/open-mmlab/mmediting/blob/main/configs/mvit/metafile.yml). In particular, you may have to refactor this README into a standard one. [Example](https://github.com/open-mmlab/mmediting/blob/main/configs/swin_transformer/README.md) -->
