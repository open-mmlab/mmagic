# Installation

We highly recommend developers follow our best practices to install MMEditing.
However, the whole process is highly customizable.
See [Customize Installation](#customize-installation) section for more information.

## Best Practices

The following steps work on Linux, Windows, and macOS.
If you have already set up a PyTorch environment, no matter using conda or pip, you can start from **step 3**.

**Step 0.**
Download and install Miniconda from [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 1.**
Create a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html#) and activate it

```shell
conda create --name mmedit python=3.8 -y
conda activate mmedit
```

**Step 2.**
Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

- On GPU platforms:

  ```shell
  conda install pytorch=1.10 torchvision cudatoolkit=11.3 -c pytorch
  ```

- On CPU platforms:

  ```shell
  conda install pytorch=1.10 torchvision cpuonly -c pytorch
  ```

**Step 3.**
Install pre-built [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip3 install openmim
mim install mmcv-full==1.5.0
```

**Step 4.**
Install [MMEditing](https://github.com/open-mmlab/mmediting) from the source code.

```shell
git clone https://github.com/open-mmlab/mmediting.git
cd mmediting
pip3 install -e .
```

**Step 5.**
Verification.

```shell
cd ~
python -c "import mmedit; print(mmedit.__version__)"
# Example output: 0.14.0
```

The installation is successful if the version number is output correctly.

## Customize Installation

### Version of Dependencies

You may change the version of Python, PyTorch, and MMCV by changing the version numbers in **Step 1, 2, and 3**, respectively.

Currently, MMEditing works with

- Python >= 3.6
- [PyTorch](https://pytorch.org/) >= 1.5
- [MMCV](https://github.com/open-mmlab/mmcv) >= 1.3.13

### Version of CUDA

When installing PyTorch in **Step 2**, you need to specify the version of CUDA.
If you are not clear on which to choose, follow our recommendations:

1. For Ampere-based NVIDIA GPUs, such as GeForce 30 series and NVIDIA A100, **CUDA 11 is a must**.
2. For older NVIDIA GPUs, CUDA 11 is backward compatible, but CUDA 10.2 offers better compatibility and is more lightweight.

Please also make sure the GPU driver satisfies the minimum version requirements.
See [this table](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions) for more information.

Please **Note** that there is no need to install the complete CUDA toolkit if you follow our [best practices](#best-practices) because no CUDA code will be compiled.
However, if you hope to compile MMCV or other C++/CUDA operators, you need to install the complete CUDA toolkit from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads), and **its version should match the CUDA version of PyTorch**, which is the version of `cudatoolkit` in `conda install`.

### Install without Conda

Though we highly recommend using conda to create environments and install PyTorch, it is viable to install PyTorch only with pip, for example, with the following command,

```shell
pip3 install torch torchvision
```

However, an `--extra-index-url` or `-f` option is usually required to specify the CPU / CUDA version.
See [PyTorch website](https://pytorch.org/get-started/locally/) for more details.

### Install without MIM

[MMCV](https://github.com/open-mmlab/mmcv) contains C++ and CUDA extensions, thus depending on PyTorch in a complex way.
[MIM](https://github.com/open-mmlab/mim) solves such dependency automatically and makes installation easier.
However, it is not a must.

To install MMCV with `pip` instead of MIM, please follow [MMCV installation guides](https://mmcv.readthedocs.io/en/latest/get_started/installation.html).
This requires manually specifying a *find-url* based on PyTorch version and its CUDA version.

For example, the following command install `mmcv-full` built for PyTorch 1.10.x and CUDA 11.3.

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
```

### On macOS

Pre-built MMCV package is not available for macOS, so you have to build MMCV from the source.
Pip will build it automatically during installation, but it requires a C++ compiler.
A simple solution is to install Clang with `xcode-select --install`.

Under such circumstances, MIM is not required and `pip install mmcv-full -v` can do the job.
See [MMCV installation guides](https://mmcv.readthedocs.io/en/latest/get_started/build.html) for more details.

### On Google Colab

Online machine learning platform such as [Google Colab](https://research.google.com/) usually has PyTorch installed.
Thus we only need to install MMCV and MMEditing with the following commands.

**Step 1.**
Install pre-built [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```ipython
!pip3 install openmim
!mim install mmcv-full
```

**Step 2.**
Install MMEditing from the source.

```ipython
!git clone https://github.com/open-mmlab/mmediting.git
%cd mmediting
!pip3 install -e .
```

**Step 3.**
Verification.

```python
import mmedit
print(mmedit.__version__)
# Example output: 0.13.0
```

**Note**: within Jupyter, the exclamation mark `!` is used to call external executables and `%cd` is a [IPython magic command](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-cd) to change the current working directory of Python.

## Additional Notes

### Speed up Installation with Mirror

One can configure conda and pip mirrors to speed up the installation.
This step is very practical for users geologically in China.

See the below links (in Chinese) for details：

- <https://mirrors.tuna.tsinghua.edu.cn/help/pypi/>
- <https://mirror.tuna.tsinghua.edu.cn/help/anaconda/>
- <https://developer.aliyun.com/mirror/pypi>

### On `-e .`

You may be curious about what `-e .` means when supplied with `pip install`.
Here is the description:

- `-e` means [editable mode](https://pip.pypa.io/en/latest/cli/pip_install/#cmdoption-e). When `import mmedit`, modules under the cloned directory are imported. If `pip install` without `-e`, pip will copy cloned codes to somewhere like `lib/python/site-package`. Consequently, modified code under the cloned directory takes no effect unless `pip install` again. This is particularly convenient for developers. If some codes are modified, new codes will be imported next time without reinstallation.
- `.` means code in this directory

You can also use `pip -e .[all]`, which will install more dependencies, especially for pre-commit hooks and unittests.
