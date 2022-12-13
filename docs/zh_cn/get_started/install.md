```{warning}
中文版本的安装文档过于陈旧，请参考英文版本。
如果您希望帮助翻译英文版安装文档，请通过issue联系我们
```

## 依赖

- Linux / Windows / Mac
- Python 3.6+
- PyTorch 1.5 或更高
- CUDA 9.0 或更高
- NCCL 2
- GCC 5.4 或更高
- [mmcv](https://github.com/open-mmlab/mmcv)

## 安装

a. 创建并激活 conda 虚拟环境，如 `python 3.8`：

```shell
conda create -n mmedit python=3.8 -y
conda activate mmedit
```

b. 按照 [PyTorch 官方文档](https://pytorch.org/) 安装 PyTorch 和 torchvision，然后安装对应路径下的 `mmcv-full`

如 `cuda 10.1` & `pytorch 1.7`：

```shell
conda install pytorch==1.7.1 torchvision cudatoolkit=10.1 -c pytorch
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7/index.html "opencv-python<=4.5.4.60"
```

注 1：过高版本的 `opencv-python` 在使用中存在一些问题，因此在安装时限制其版本。

注 2：确保 CUDA 编译版本和 CUDA 运行版本相匹配。 用户可以参照 [PyTorch 官网](https://pytorch.org/) 对预编译包所支持的 CUDA 版本进行核对。

`例1`：如果 `/usr/local/cuda` 文件夹下已安装了 CUDA 10.1 版本，则需要安装 CUDA 10.1 下预编译的 PyTorch。

```shell
conda install pytorch cudatoolkit=10.1 torchvision -c pytorch
```

`例2`：如果你在 `/usr/local/cuda` 文件夹下已安装 CUDA 9.2 版本，并且想要安装 PyTorch 1.3.1 版本，则需要安装 CUDA 9.2 下预编译的 PyTorch。

```shell
conda install pytorch=1.3.1 cudatoolkit=9.2 torchvision=0.4.2 -c pytorch
```

c. 克隆 MMEditing 仓库

```shell
git clone https://github.com/open-mmlab/mmediting.git
cd mmediting
```

d. 安装相关依赖和 MMEditing

```shell
pip install -r requirements.txt
pip install -v -e .  # or "python setup.py develop"
```

如果你是在 macOS 环境下安装，则需将上述命令的最后一行替换为：

```shell
CC=clang CXX=clang++ CFLAGS='-stdlib=libc++' pip install -e .
```

e. 验证安装

安装完成后，可以切换到其他目录（例如 `/home` 目录），并尝试在 python 中导入 mmedit，导入成功则证明安装成功

```shell
$ cd ~
$ python

>>> import mmedit
>>> mmedit.__version__
'0.12.0'
```

注：

1. git commit 的 id 将会被写到版本号中，如 0.6.0+2e7045c。这个版本号也会被保存到训练好的模型中。 推荐用户每次在对本地代码和 github 上的源码进行同步后，执行一次步骤 b。如果 C++/CUDA 代码被修改，就必须进行这一步骤。

   > 重要：如果你在一个新的 CUDA/PyTorch 版本下重新安装了 mmedit ，请确保删除了`./build`文件夹

   ```shell
   pip uninstall mmedit
   rm -rf ./build
   find . -name "*.so" | xargs rm
   ```

2. 根据上述步骤， MMEditing 就会以 `dev` 模式被安装，任何本地的代码修改都会立刻生效，不需要再重新安装一遍（除非用户提交了 commits，并且想更新版本号）。

3. 如果用户想使用 `opencv-python-headless` 而不是 `opencv-python`，可在安装 `MMCV` 前安装 `opencv-python-headless`。

4. 有些模型（例如图像修复任务中的 `EDVR`）依赖于 `mmcv-full` 中的一些 CUDA 算子，具体的依赖可在 `requirements.txt` 中找到。
   如需要，请通过 `pip install -r requirements.txt` 命令来安装 `mmcv-full`，安装过程中会在本地编译 CUDA 算子，这个过程大概需要10分钟。
   另一种方案是安装预编译版本的 `mmcv-full`，请参考 [MMCV 主页](https://github.com/open-mmlab/mmcv#install-with-pip) 获取具体的安装指令。
   此外，如果你要使用的模型不依赖于 CUDA 算子，那么也可以使用 `pip install mmcv`来安装轻量版本的 mmcv，其中 CUDA 算子被移除了。

### CPU 环境下的安装步骤

MMEditing 也可以在只有 CPU 的环境下安装（即无法使用 GPU 的环境）。

相应的，安装 CPU 版本的 PyTorch 和 MMCV

```shell
conda install pytorch==1.7.1 torchvision cudatoolkit=10.1 -c pytorch
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7/index.html "opencv-python<=4.5.4.60"
```

然而在该环境下，有些功能将被移除，例如：

- Deformable Convolution（可变形卷积）

因此，如果您尝试使用一些包含可变形卷积的模型进行推理，则会出现错误。

### 利用 Docker 镜像安装 MMEditing

MMEditing 提供了一个 [Dockerfile](https://github.com/open-mmlab/mmediting/blob/master/docker/Dockerfile) 来创建 docker 镜像

```shell
# build an image with PyTorch 1.5, CUDA 10.1
docker build -t mmediting docker/
```

运行以下命令：

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmediting/data mmediting
```

### 完整的安装脚本

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

# install latest pytorch prebuilt with the default prebuilt CUDA version (usually the latest)
conda install -c pytorch pytorch torchvision -y
git clone https://github.com/open-mmlab/mmediting.git
cd mmediting
pip install -r requirements.txt
pip install -v -e .
```
