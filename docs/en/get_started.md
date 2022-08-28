# Get Started: Install and run MMEditing

## Prerequisite

Currently, MMEditing works with

- Python >= 3.6
- [PyTorch](https://pytorch.org/) >= 1.5
- [MMCV](https://github.com/open-mmlab/mmcv) >= 1.3.13

1. For Ampere-based NVIDIA GPUs, such as GeForce 30 series and NVIDIA A100, **CUDA 11 is a must**.
2. For older NVIDIA GPUs, CUDA 11 is backward compatible, but CUDA 10.2 offers better compatibility and is more lightweight.
3. Please also make sure the GPU driver satisfies the minimum version requirements. See [this table](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions) for more information.
4. If you hope to compile MMCV or other C++/CUDA operators, you need to install the complete CUDA toolkit from [NVIDIA&#39;s website](https://developer.nvidia.com/cuda-downloads), and **its version should match the CUDA version of PyTorch**, which is the version of `cudatoolkit` in `conda install`.


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

You may be curious about what `-e .` means when supplied with `pip install`.
Here is the description:

- `-e` means [editable mode](https://pip.pypa.io/en/latest/cli/pip_install/#cmdoption-e). When `import mmedit`, modules under the cloned directory are imported. If `pip install` without `-e`, pip will copy cloned codes to somewhere like `lib/python/site-package`. Consequently, modified code under the cloned directory takes no effect unless `pip install` again. This is particularly convenient for developers. If some codes are modified, new codes will be imported next time without reinstallation.
- `.` means code in this directory

You can also use `pip -e .[all]`, which will install more dependencies, especially for pre-commit hooks and unittests.


## Run with MMEditing

After installing MMEditing successfully, now you are able to run with MMEditing! 

Here, we provide an example of running a demo of image super-resolution. 
You can use the following commands to improve the resolution of your image.

```shell
python demo/restoration_demo.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    ${IMAGE_FILE} \
    ${SAVE_FILE} \
    [--imshow] \
    [--device ${GPU_ID}] \
    [--ref-path ${REF_PATH}]
```

If `--imshow` is specified, the demo will also show image with opencv. Examples:

```shell
python demo/restoration_demo.py \
    configs/esrgan/esrgan_x4c64b23g32_400k-1xb16_div2k.py \
    https://download.openmmlab.com/mmediting/restorers/esrgan/esrgan_x4c64b23g32_1x16_400k_div2k_20200508-f8ccaf3b.pth \
    tests/data/image/lq/baboon_x4.png \
    demo/demo_out_baboon.png
```

You can test Ref-SR by providing `--ref-path`. Examples:

```shell
python demo/restoration_demo.py \
    configs/ttsr/ttsr-gan_x4c64b16_500k-1xb9_CUFED.py \
    https://download.openmmlab.com/mmediting/restorers/ttsr/ttsr-gan_x4_c64b16_g1_500k_CUFED_20210626-2ab28ca0.pth \
    tests/data/frames/sequence/gt/sequence_1/00000000.png \
    demo/demo_out.png \
    --ref-path tests/data/frames/sequence/gt/sequence_1/00000001.png
```
