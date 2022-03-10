Installation
############

MMEditing is a Python toolbox based on `PyTorch`_ and `MMCV`_.
Currently, MMEditing works on Linux, Windows, macOS, and on both CPUs and GPUs.
This page describes some best practices for installation.

If you have difficulties installing MMEditing, please feel free to `open a discussion <https://github.com/open-mmlab/mmediting/discussions>`_ and describe the error as well as the steps you have tried.
The community are happy to help.


Prerequisites
=============

* Linux (recommended) / Windows / macOS
* `Python`_ >= 3.6
* `PyTorch`_ >= 1.5
* `pip`_ and `conda`_
* `Git`_
* (Only for GPU) NVIDIA GPU with `driver`_ version >= 440.33 (Linux) or >= 441.22 (Windows native)
* (Only for macOS) Clang compiler, can be installed with ``xcode-select -–install``


.. warning::

   As Python 3.6 has reached `end-of-life`_ on 23-Dec-2021, we will drop support for it in the future.


Install CPU Version
===================

MMEditing is fully supported on CPUs, despite the slow running speed.
Nevertheless, the CPU version is much more lightweight and easier to configure compared to GPU versions.

For macOS, this is the only choice.

**Step 1**.
Create and activate a conda virtual environment

.. code-block:: sh

   conda create --name mmedit python=3.8 -y
   conda activate mmedit


**Step 2**.
Install the CPU version of PyTorch and torchvision

.. code-block:: sh

   conda install pytorch torchvision cpuonly -c pytorch

   # or "pip install torch torchvision"


If you prefer to install previous versions of PyTorch, follow `this guide <https://pytorch.org/get-started/previous-versions/>`_.

If PyTorch is already installed, check its version with:

.. code-block:: sh

   python -c "import torch; print(torch.__version__)"
   # Example output: 1.10.2


**Step 3**.
Install pre-compiled MMCV for CPUs

.. code-block:: sh

   pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.10/index.html


.. note::

   The link to MMCV depends on the version of PyTorch.
   The above link is for **PyTorch 1.10.x**.
   If you use a different version of PyTorch, you need to change the link accordingly.

   E.g. ``https://download.openmmlab.com/mmcv/dist/cpu/`` **torch1.9** ``/index.html`` for **PyTorch 1.9.x**.

.. note::

   Precompiled packages are available for Linux and Windows only.
   On macOS, install MMCV with the following command.

   .. code-block::

      CC=clang CXX=clang++ CFLAGS='-stdlib=libc++' MMCV_WITH_OPS=1 pip install -e .

   See `MMCV documentation <https://mmcv.readthedocs.io/en/latest/get_started/build.html#build-on-linux-or-macos>`_
   for more information.

   A fallback option is to install the lite version of MMCV via ``pip install mmcv``.
   However, it makes *deformable convolution* unavailable, and several models won't work.


See `MMCV installation guide`_ for more information.


**Step 4**.
Install MMEditing

To make full utilization of configuration files and useful tools,
we recommend installing MMEditing from source codes.

.. code-block:: sh

   git clone https://github.com/open-mmlab/mmediting.git
   cd mmediting
   pip install -v -e .

   # or "pip install -v -e .[all]" to install full dependencies and enable more features


**Step 5**.
Verify installation

As a simple test, switch to another directory (such as ``/home``) and import ``mmedit`` in Python.
For example,

.. code-block:: sh

   cd ~
   python -c "import mmedit; print(mmedit.__version__)"
   # Example output: 0.13.0

Make sure the version outputs correctly.
For comprehensive unit tests, run ``pytest .``.


Install CUDA Version
====================

To enable the full power of MMEditing, we recommend the GPU version.
The only difference lies at PyTorch and MMCV.
Please pay attention to the **version** of and the **CUDA version** of PyTorch.

.. note::

   GPU is not available for macOS.

**Step 1**.
Create and activate a conda virtual environment

.. code-block:: sh

   conda create --name mmedit python=3.8 -y
   conda activate mmedit


**Step 2**.
Install the GPU version of PyTorch and torchvision

.. code-block:: sh

   conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

   # or "pip install torch==1.10.2+cu102 torchvision==0.11.3+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html"


.. note::

   To support new GPU models such as GTX 30 series, CUDA 11 is required. Install CUDA-11 based PyTorch with:

   .. code-block:: sh

      conda install pytorch torchvision cudatoolkit=11.3 -c pytorch

      # or pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

   Please check `this table`_ for minimum driver versions required for specific CUDA versions.
   Usually, the latest driver works well.


If you prefer to install previous versions of PyTorch, follow `this guide <https://pytorch.org/get-started/previous-versions/>`_.


If PyTorch is already installed, check its *version* and *CUDA version* in Python:

.. code-block:: sh

   python -c "import torch; print(torch.__version__)"
   # Example output: 1.10.2
   python -c "import torch; print(torch.version.cuda)"
   # Example output: 10.2


**Step 3**.
Install pre-compiled MMCV for GPUs

.. code-block:: sh

   pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10/index.html

.. note::

   The link to MMCV depends on the *version* of PyTorch and the *CUDA version* of PyTorch.
   The above link is for **PyTorch 1.10.x** and **CUDA 10.2**.
   If you use a different version of PyTorch, you need to change the link accordingly.

   E.g. ``https://download.openmmlab.com/mmcv/dist/`` **cu101** ``/`` **torch1.8** ``/index.html`` for **PyTorch 1.8.x** with CUDA 10.1.

See `MMCV installation guide`_ for more information.


**Step 4**.
Install MMEditing

To make full utilization of configuration files and useful tools,
we recommend installing MMEditing from source codes.

.. code-block:: sh

   git clone https://github.com/open-mmlab/mmediting.git
   cd mmediting
   pip install -v -e .

   # or "pip install -v -e .[all]" to install full dependencies and enable more features


**Step 5**.
Verify installation

As a simple test, switch to another directory (such as ``/home``) and import ``mmedit`` in Python.
For example,

.. code-block:: sh

   cd ~
   python -c "import mmedit; print(mmedit.__version__)"
   # Example output: 0.13.0

Make sure the version outputs correctly.
For comprehensive unit tests, run ``pytest .``.


Another option: Install via MIM
===============================

MMEditing can be installed via `MIM`_, a package manager dedicated to OpenMMLab projects.
See `MIM documentations`_ for instructions.


Another option: Docker Image
============================

We provide a `Dockerfile <https://github.com/open-mmlab/mmediting/blob/master/docker/Dockerfile>`_ for building a docker image.

To build the image:

.. code-block:: sh

   # build an image with PyTorch 1.5, CUDA 10.1
   docker build -t mmediting docker/


Run with:

.. code-block:: sh

   docker run --gpus all --shm-size=8g -it -v ${DATA_DIR}:/mmediting/data mmediting



After installation, you can run some demos, click next.


.. _Git: https://git-scm.com/
.. _Python: https://www.python.org/
.. _conda: https://docs.conda.io/en/latest/
.. _pip: https://pip.pypa.io/en/stable/
.. _MMCV: https://github.com/open-mmlab/mmcv
.. _PyTorch: https://pytorch.org/
.. _end-of-life: https://endoflife.date/python
.. _NVIDIA driver: https://www.nvidia.com/download/index.aspx
.. _driver: https://www.nvidia.com/download/index.aspx
.. _this table: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions
.. _PyTorch installation guide: https://pytorch.org/get-started/locally/
.. _MMCV installation guide: https://mmcv.readthedocs.io/en/latest/get_started/installation.html
.. _MIM: https://github.com/open-mmlab/mim
.. _MIM documentations: https://openmim.readthedocs.io/en/latest/index.html
.. _WSL_CUDA: https://docs.nvidia.com/cuda/wsl-user-guide/index.html
.. _WSL: https://docs.microsoft.com/en-us/windows/wsl/install
