## Installation

### Requirements

- Linux (Windows is not officially supported)
- Python 3.6+
- PyTorch 1.3 or higher
- CUDA 9.0 or higher
- NCCL 2
- GCC 4.9 or higher
- [mmcv](https://github.com/open-mmlab/mmcv)

### Install mmediting

a. Create a conda virtual environment and activate it.

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch torchvision -c pytorch
```

Note: Make sure that your compilation CUDA version and runtime CUDA version match.
You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

`E.g. 1` If you have CUDA 10.1 installed under `/usr/local/cuda` and would like to install
PyTorch 1.5, you need to install the prebuilt PyTorch with CUDA 10.1.

```python
conda install pytorch cudatoolkit=10.1 torchvision -c pytorch
```

`E.g. 2` If you have CUDA 9.2 installed under `/usr/local/cuda` and would like to install
PyTorch 1.3.1., you need to install the prebuilt PyTorch with CUDA 9.2.

```python
conda install pytorch=1.3.1 cudatoolkit=9.2 torchvision=0.4.2 -c pytorch
```

If you build PyTorch from source instead of installing the prebuilt pacakge,
you can use more CUDA versions such as 9.0.

c. Clone the mmediting repository.

```shell
git clone https://github.com/open-mmlab/mmediting.git
cd mmediting
```

d. Install build requirements and then install mmediting.

```shell
pip install -r requirements.txt
pip install -v -e .  # or "python setup.py develop"
```

If you build mmediting on macOS, replace the last command with

```
CC=clang CXX=clang++ CFLAGS='-stdlib=libc++' pip install -e .
```

Note:

1. The git commit id will be written to the version number with step d, e.g. 0.6.0+2e7045c. The version will also be saved in trained models.
It is recommended that you run step d each time you pull some updates from github. If C++/CUDA codes are modified, then this step is compulsory.

    > Important: Be sure to remove the `./build` folder if you reinstall mmedit with a different CUDA/PyTorch version.

    ```
    pip uninstall mmedit
    rm -rf ./build
    find . -name "*.so" | xargs rm
    ```

2. Following the above instructions, mmediting is installed on `dev` mode, any local modifications made to the code will take effect without the need to reinstall it (unless you submit some commits and want to update the version number).

3. If you would like to use `opencv-python-headless` instead of `opencv-python`,
you can install it before installing MMCV.

4. Some models (such as EDVR in restorers) depend on CUDA ops in `mmcv-full` which is listed in `requirements.txt`. Install it with the default command `pip install -r requirements.txt` need to compile CUDA ops locally and it may take up to 10 mins. Another option is to install pre-compiled `mmcv-full`, visit [MMCV github page](https://github.com/open-mmlab/mmcv#install-with-pip) for concrete instructions. Moreover, if the model you intend to use does not depend on CUDA ops, you could also install the lite version of mmcv with `pip install mmcv` in which CUDA ops is excluded.

### Install with CPU only
The code can be built for CPU only environment (where CUDA isn't available).

<!-- In CPU mode you can run the demo/webcam_demo.py for example. -->
However some functionality is gone in this mode:

- Deformable Convolution

So if you try to run inference with a model containing deformable convolution you will get an error.

### Another option: Docker Image

We provide a [Dockerfile](https://github.com/open-mmlab/mmediting/blob/master/docker/Dockerfile) to build an image.

```shell
# build an image with PyTorch 1.5, CUDA 10.1
docker build -t mmediting docker/
```

Run it with

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmediting/data mmediting
```

### A from-scratch setup script

Here is a full script for setting up mmediting with conda.

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

<!-- ### Using multiple MMEditing versions

The train and test scripts already modify the `PYTHONPATH` to ensure the script use the MMEditing in the current directory.

To use the default MMEditing installed in the environment rather than that you are working with, you can remove the following line in those scripts

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
``` -->
