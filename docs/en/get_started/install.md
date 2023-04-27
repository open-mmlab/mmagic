# Installation

In this section, you will know about:

- [Installation](#installation)
  - [Installation](#installation-1)
    - [Prerequisites](#prerequisites)
    - [Best practices](#best-practices)
    - [Customize installation](#customize-installation)
      - [CUDA Version](#cuda-version)
      - [Install MMCV without MIM](#install-mmcv-without-mim)
      - [Using MMagic with Docker](#using-mmagic-with-docker)
      - [Trouble shooting](#trouble-shooting)
    - [Developing with multiple MMagic versions](#developing-with-multiple-mmagic-versions)

## Installation

We recommend that users follow our [Best practices](#best-practices) to install MMagic.
However, the whole process is highly customizable. See [Customize installation](#customize-installation) section for more information.

### Prerequisites

In this section, we demonstrate how to prepare an environment with PyTorch.

MMagic works on Linux, Windows, and macOS. It requires:

- Python >= 3.7
- [PyTorch](https://pytorch.org/) >= 1.8
- [MMCV](https://github.com/open-mmlab/mmcv) >= 2.0.0

>

If you are experienced with PyTorch and have already installed it,
just skip this part and jump to the [next section](#best-practices). Otherwise, you can follow these steps for the preparation.

**Step 0.**
Download and install Miniconda from [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 1.**
Create a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html#) and activate it

```shell
conda create --name mmagic python=3.8 -y
conda activate mmagic
```

**Step 2.**
Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

- On GPU platforms:

  ```shell
  conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
  ```

- On CPU platforms:

  ```shell
  conda install pytorch=1.10 torchvision cpuonly -c pytorch
  ```

### Best practices

**Step 0.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install 'mmcv>=2.0.0'
```

**Step 1.** Install [MMEngine](https://github.com/open-mmlab/mmengine).

```shell
mim install 'mmengine'
```

Or

```shell
pip install mmengine
```

Or

```shell
pip install git+https://github.com/open-mmlab/mmengine.git
```

**Step 2.** Install MMagic.

```shell
mim install 'mmagic'
```

Or

```shell
pip install mmagic
```

Or install [MMagic](https://github.com/open-mmlab/mmagic) from the source code.

```shell
git clone https://github.com/open-mmlab/mmagic.git
cd mmagic
pip3 install -e . -v
```

**Step 5.**
Verify MMagic has been successfully installed.

```shell
cd ~
python -c "import mmagic; print(mmagic.__version__)"
# Example output: 1.0.0
```

The installation is successful if the version number is output correctly.

```{note}
You may be curious about what `-e .` means when supplied with `pip install`.
Here is the description:

- `-e` means [editable mode](https://pip.pypa.io/en/latest/cli/pip_install/#cmdoption-e).
  When `import mmagic`, modules under the cloned directory are imported.
  If `pip install` without `-e`, pip will copy cloned codes to somewhere like `lib/python/site-package`.
  Consequently, modified code under the cloned directory takes no effect unless `pip install` again.
  Thus, `pip install` with `-e` is particularly convenient for developers. If some codes are modified, new codes will be imported next time without reinstallation.
- `.` means code in this directory

You can also use `pip install -e .[all]`, which will install more dependencies, especially for pre-commit hooks and unittests.
```

### Customize installation

#### CUDA Version

When installing PyTorch, you need to specify the version of CUDA. If you are not clear on which to choose, follow our recommendations:

- For Ampere-based NVIDIA GPUs, such as GeForce 30 series and NVIDIA A100, CUDA 11 is a must.
- For older NVIDIA GPUs, CUDA 11 is backward compatible, but CUDA 10.2 offers better compatibility and is more lightweight.

Please make sure the GPU driver satisfies the minimum version requirements.
See [this table](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions) for more information.

**note**
Installing CUDA runtime libraries is enough if you follow our best practices,
because no CUDA code will be compiled locally.
However, if you hope to compile MMCV from source or develop other CUDA operators,
you need to install the complete CUDA toolkit from NVIDIA's [website](https://developer.nvidia.com/cuda-downloads),
and its version should match the CUDA version of PyTorch. i.e., the specified version of cudatoolkit in `conda install` command.

#### Install MMCV without MIM

MMCV contains C++ and CUDA extensions, thus depending on PyTorch in a complex way.
MIM solves such dependencies automatically and makes the installation easier. However, it is not a must.

To install MMCV with pip instead of MIM, please follow [MMCV installation guides](https://mmcv.readthedocs.io/en/latest/get_started/installation.html).
This requires manually specifying a find-url based on PyTorch version and its CUDA version.

For example, the following command install mmcv-full built for PyTorch 1.10.x and CUDA 11.3.

```shell
pip install 'mmcv>=2.0.0' -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
```

#### Using MMagic with Docker

We provide a [Dockerfile](https://github.com/open-mmlab/mmagic/blob/main/docker/Dockerfile) to build an image.
Ensure that your [docker version](https://docs.docker.com/engine/install/) >=19.03.

```shell
# build an image with PyTorch 1.8, CUDA 11.1
# If you prefer other versions, just modified the Dockerfile
docker build -t mmagic docker/
```

Run it with

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmagic/data mmagic
```

#### Trouble shooting

If you have some issues during the installation, please first view the [FAQ](../faq.md) page.
You may [open an issue](https://github.com/open-mmlab/mmagic/issues/new/choose) on GitHub if no solution is found.

### Developing with multiple MMagic versions

The train and test scripts already modify the `PYTHONPATH` to ensure the script uses the `MMagic` in the current directory.

To use the default MMagic installed in the environment rather than that you are working with, you can remove the following line in those scripts

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```
