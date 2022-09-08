# Get Started: Install and Run MMEditing

## Prerequisite

Currently, MMEditing works with

- Python >= 3.6
- [PyTorch](https://pytorch.org/) >= 1.5
- [MMCV](https://github.com/open-mmlab/mmcv) >= 2.0.0rc1

1. For Ampere-based NVIDIA GPUs, such as GeForce 30 series and NVIDIA A100, **CUDA 11 is a must**.
2. For older NVIDIA GPUs, CUDA 11 is backward compatible, but CUDA 10.2 offers better compatibility and is more lightweight.
3. Please also make sure the GPU driver satisfies the minimum version requirements. See [this table](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions) for more information.
4. If you hope to compile MMCV or other C++/CUDA operators, you need to install the complete CUDA toolkit from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads), and **its version should match the CUDA version of PyTorch**, which is the version of `cudatoolkit` in `conda install`.

## Installation

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
  conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
  ```

- On CPU platforms:

  ```shell
  conda install pytorch torchvision cpuonly -c pytorch
  ```

**Step 3.**
Install pre-built [MMCV](https://github.com/open-mmlab/mmcv) and [MMEngine](https://github.com/open-mmlab/mmengine) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip3 install openmim
mim install 'mmcv>=2.0.0rc1'
pip install git+https://github.com/open-mmlab/mmengine.git
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

You may be curious about what `-e .` means when supplied with `pip install`.
Here is the description:

- `-e` means [editable mode](https://pip.pypa.io/en/latest/cli/pip_install/#cmdoption-e). When `import mmedit`, modules under the cloned directory are imported. If `pip install` without `-e`, pip will copy cloned codes to somewhere like `lib/python/site-package`. Consequently, modified code under the cloned directory takes no effect unless `pip install` again. Thus, `pip install` with `-e` is particularly convenient for developers. If some codes are modified, new codes will be imported next time without reinstallation.
- `.` means code in this directory

You can also use `pip install -e .[all]`, which will install more dependencies, especially for pre-commit hooks and unittests.

## Quick Run

After installing MMEditing successfully, now you are able to run with MMEditing!

Here, we provide an example of running a demo of image super-resolution.
To make your photos much more clear, you only need several lines of codes by MMEditing!

```python
import mmcv
from mmedit.apis import init_model, restoration_inference
from mmedit.engine.misc import tensor2img

config = 'configs/esrgan/esrgan_x4c64b23g32_1xb16-400k_div2k.py'
checkpoint = 'https://download.openmmlab.com/mmediting/restorers/esrgan/esrgan_x4c64b23g32_1x16_400k_div2k_20200508-f8ccaf3b.pth'
img_path = 'tests/data/image/lq/baboon_x4.png'
model = init_model(config, checkpoint)
output = restoration_inference(model, img_path)
output = tensor2img(output)
mmcv.imwrite(output, 'output.png')
```

Now, you can check your fancy photo in `output.png`.
