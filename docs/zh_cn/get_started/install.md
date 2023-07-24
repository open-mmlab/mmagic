# 安装教程

在本节中，你将了解到：

- [安装教程](#安装教程)
  - [安装](#安装)
    - [前提条件](#前提条件)
    - [最佳实践](#最佳实践)
    - [自定义安装](#自定义安装)
      - [CUDA版本](#cuda版本)
      - [不使用MIM安装MMCV](#不使用mim安装mmcv)
      - [在Docker中使用MMagic](#在docker中使用mmagic)
      - [问题解决](#问题解决)
    - [使用多个MMagic版本开发](#使用多个mmagic版本开发)

## 安装

我们建议用户按照我们的[最佳实践](#最佳实践)来安装MMagic。
然而，整个过程是高度可定制的。更多信息请参阅[自定义安装](#自定义安装)部分。

### 前提条件

在本节中，我们将演示如何使用PyTorch准备环境。

MMagic可以在Linux, Windows, 和macOS上运行。它要求：

- Python >= 3.7
- [PyTorch](https://pytorch.org/) >= 1.8
- [MMCV](https://github.com/open-mmlab/mmcv) >= 2.0.0

>

如果您对PyTorch有经验并且已经安装了它，直接跳过这一部分，跳到[下一节](#最佳实践)。否则, 您可以按照以下步骤来准备环境。

**Step 0.**
从[官方网站](https://docs.conda.io/en/latest/miniconda.html)下载和安装Miniconda.

**Step 1.**
创建一个[conda虚拟环境](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html#)并激活它

```shell
conda create --name mmagic python=3.8 -y
conda activate mmagic
```

**Step 2.**
按照[官方说明](https://pytorch.org/get-started/locally/)安装PyTorch，例如

- 在GPU平台上：

  ```shell
  conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
  ```

- 在CPU平台上：

  ```shell
  conda install pytorch=1.10 torchvision cpuonly -c pytorch
  ```

### 最佳实践

**Step 0.** 使用[MIM](https://github.com/open-mmlab/mim)安装[MMCV](https://github.com/open-mmlab/mmcv)。

```shell
pip install -U openmim
mim install 'mmcv>=2.0.0'
```

**Step 1.** 安装[MMEngine](https://github.com/open-mmlab/mmengine)。

```shell
mim install 'mmengine'
```

或者

```shell
pip install mmengine
```

或者

```shell
pip install git+https://github.com/open-mmlab/mmengine.git
```

**Step 2.** 安装MMagic。

```shell
mim install 'mmagic'
```

或者

```shell
pip install mmagic
```

或者从源代码安装[MMagic](https://github.com/open-mmlab/mmagic)。

```shell
git clone https://github.com/open-mmlab/mmagic.git
cd mmagic
pip3 install -e . -v
```

**Step 5.**
检查MMagic是否安装成功。

```shell
cd ~
python -c "import mmagic; print(mmagic.__version__)"
# 示例输出: 1.0.0
```

显示正确的版本号，则表示安装成功。

```{note}
你可能想知道附加在`pip install`后面的`-e .`是什么意思。
下面是说明:

- `-e`表示[可编辑模式](https://pip.pypa.io/en/latest/cli/pip_install/#cmdoption-e).
  当`import mmagic`时，将导入克隆目录下的模块。
  如果`pip install`没有附加`-e`, pip会将克隆的代码复制到类似`lib/python/site-package`的地方。
  因此，除非再次执行`pip install`命令，否则在克隆目录下修改后的代码不会生效。
  因此，`pip install`命令附带`-e`对于开发人员来说特别方便。如果修改了一些代码，下次导入新的代码时不需要重新安装。
- `.`表示此目录中的代码。

你也可以使用`pip install -e .[all]`命令，这将安装更多的依赖项，特别是对于预提交hooks和单元测试。
```

### 自定义安装

#### CUDA版本

安装PyTorch时,您需要指定CUDA的版本。如果您不清楚该选择哪一个，请遵循我们的建议:

- 对于基于Ampere的NVIDIA GPUs，如GeForce 30系列和NVIDIA A100，必须使用CUDA 11。
- 对于较老的NVIDIA GPUs，是向后兼容的，但CUDA 10.2提供了更好的兼容性，更轻量。

请确保GPU驱动程序满足最低版本要求。
更多信息请参见[此表](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions)。

**注意**
如果遵循我们的最佳实践，安装CUDA runtime库就足够了，因为不会在本地编译CUDA代码。
但是，如果您希望从源代码编译MMCV或开发其他CUDA算子，则需要从NVIDIA的[开发者网站](https://developer.nvidia.com/cuda-downloads)安装完整的CUDA工具包，其版本应与PyTorch的CUDA版本匹配。即，在 `conda install` 命令中指定的cudatoolkit版本。

#### 不使用MIM安装MMCV

MMCV包含c++和CUDA扩展，因此以一种复杂的方式依赖于PyTorch。MIM自动解决了这种依赖关系，并使安装更容易。然而，这并不是必须的。

要使用pip而不是MIM安装MMCV，请遵循[MMCV安装指南](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)。这需要根据PyTorch版本及其CUDA版本手动指定find-url。

例如，以下命令install mmcv-full是针对PyTorch 1.10.x和CUDA 11.3构建的。

```shell
pip install 'mmcv>=2.0.0' -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
```

#### 在Docker中使用MMagic

我们提供一个[Dockerfile](https://github.com/open-mmlab/mmagic/blob/main/docker/Dockerfile)来构建一个镜像。请确保您的[docker版本](https://docs.docker.com/engine/install/)>=19.03。

```shell
# 使用PyTorch 1.8, CUDA 11.1构建一个镜像
# 如果您喜欢其他版本，只需修改Dockerfile
docker build -t mmagic docker/
```

使用如下命令运行

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmagic/data mmagic
```

#### 问题解决

如果在安装过程中遇到问题，请先查看[FAQ](../faq.md)页面。如果找不到解决方案，可以在GitHub上[open an issue](https://github.com/open-mmlab/mmagic/issues/new/choose)。

### 使用多个MMagic版本开发

训练和测试脚本已经修改了`PYTHONPATH`，以确保脚本使用当前目录中的`MMagic`。

要使用环境中安装的默认MMagic，而不是您正在使用的MMagic，可以删除这些脚本中的以下行

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```
