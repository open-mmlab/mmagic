Installation
############

MMEditing is a Python toolbox based on `PyTorch`_ and `MMCV`_.
These packages can be installed in various ways.
Here we describe some best practices.

If you have difficulties installing MMEditing, please feel free to `open a discussion <https://github.com/open-mmlab/mmediting/discussions>`_ and describe your situations. The community are happy to help.


Prerequisites
=============

* `Python`_ >= 3.6
* `pip`_ and/or `conda`_
* `Git`_
* (Recommended) NVIDIA GPU with driver version >= 440.33 (Linux) or >= 441.22 (Windows)
* (Optional) CUDA and C++ (GCC / Clang / MSVC) compilers if you hope to compile `MMCV`_ from source codes

.. warning::

   As Python 3.6 has reached end-of-life on 23-Dec-2021, we will drop its support in the future.


Install CPU Version
===================

MMEditing is fully supported on CPUs, despite the slow running speed.
Compared to GPU versions, the CPU version is much more lightweight and easy to configure.
So if you want to perform a quick run, the CPU version is good enough.


Step 1.
Create and activate a conda virtual environment

.. code-block:: sh

   conda create --name mmedit python=3.8 -y
   conda activate mmedit


Step 2.
Install the CPU version of PyTorch and torchvision

.. code-block:: sh

   conda install pytorch torchvision cpuonly -c pytorch

OR

.. code-block:: sh

   pip install torch torchvision

See `PyTorch installation guide <https://pytorch.org/get-started/locally/>`_ for more information.


Step 3.
Install pre-compiled MMCV for CPU environment

.. code-block:: sh

   pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.10/index.html


.. note::

   The download link of MMCV depends on the version of PyTorch.
   The above link is for **PyTorch 1.10.x**.
   If you use a different version of PyTorch, you need to change the link accordingly.

   E.g. ``https://download.openmmlab.com/mmcv/dist/cpu/`` **torch1.9** ``/index.html`` for **PyTorch 1.9.x**.

.. note::

   Precompiled package are available for Windows and Linux only.
   On macOS, you need to compile MMCV from source.
   See `instructions <https://mmcv.readthedocs.io/en/latest/get_started/build.html#build-on-linux-or-macos>`_.

   A fallback options is to install the lite version of MMCV.
   But *deformable convolution* becomes unavailable.
   Several models in MMEditnig won't run correctly.


See `MMCV documentations <https://mmcv.readthedocs.io/en/latest/get_started/installation.html>`_ for more information.


Step 4.
Clone the MMEditing repository

.. code-block:: sh

   git clone https://github.com/open-mmlab/mmediting.git
   cd mmediting


Step 5.
Install MMEditing from source codes

.. code-block:: sh

   pip install -v -e .
   # or "pip install -v -e .[all]"
   # to install full dependencies to enable more features


Step 6.
Verify installation

As a simple test, you can switch to another directory (such as ``/home``) and import ``mmedit`` in Python. For example:

.. code-block:: python

   $ cd ~
   $ python
   Python 3.7.11 (default, Jul 27 2021, 09:42:29) [MSC v.1916 64 bit (AMD64)] :: Anaconda, Inc. on win32
   Type "help", "copyright", "credits" or "license" for more information.
   >>> import mmedit
   >>> mmedit.__version__
   '0.13.0'

The installation is successful if the version number is correctly output.
For comprehensive unit tests, you can run ``pytest .``


Install CUDA Version
====================

To enable the full power of MMEditing, we recommend installing the GPU version.
Differences on installation lies at PyTorch and MMCV.
Please pay more attention to the version and the CUDA version of PyTorch and MMCV.


Step 1.
Create and activate a conda virtual environment

.. code-block:: sh

   conda create --name mmedit python=3.8 -y
   conda activate mmedit


Step 2.
Install the GPU version of PyTorch and torchvision

.. code-block:: sh

   conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

or

.. code-block:: sh

   pip3 install torch==1.10.2+cu102 torchvision==0.11.3+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html

Here, CUDA 10.2 is just an example. Other version works too.
See `PyTorch installation guide <https://pytorch.org/get-started/locally/>`_ for more options.

.. note::

   Make sure the version of GPU driver is sufficnet enough to support the specific CUDA version.
   See `CUDA driver version`_ for more information.
   Usually, the latest GPU driver works well.


Step 3.
Install pre-compiled MMCV for GPU environment

.. code-block:: sh

   pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10/index.html

.. note::

   The download link of MMCV depends on the version of PyTorch and the CUDA version of PyTorch.
   The above link is for **PyTorch 1.10.x** and **CUDA 10.2**.
   If you use a different version of PyTorch, you need to change the link accordingly.

   E.g. ``https://download.openmmlab.com/mmcv/dist/`` **cu101** ``/`` **torch1.8** ``/index.html`` for **PyTorch 1.8.x** and CUDA 10.1.

See `MMCV documentations <https://mmcv.readthedocs.io/en/latest/get_started/installation.html>`_ for more information.


Step 4.
Clone the MMEditing repository

.. code-block:: sh

   git clone https://github.com/open-mmlab/mmediting.git
   cd mmediting


Step 5.
Install MMEditing from source codes

.. code-block:: sh

   pip install -v -e .
   # or "pip install -v -e .[all]"
   # to install full dependencies for more features


Step 6.
Verify installation

As a simple test, you can switch to another directory (such as ``/home``) and import ``mmedit`` in Python.

.. code-block:: sh

   $ cd ~
   $ python
   Python 3.7.11 (default, Jul 27 2021, 09:42:29) [MSC v.1916 64 bit (AMD64)] :: Anaconda, Inc. on win32
   Type "help", "copyright", "credits" or "license" for more information.
   >>> import mmedit
   >>> mmedit.__version__
   '0.13.0'

If the version number is correctly printed, the installation is successful.
For comprehensive unittests, you can run ``pytest .``


Install via MIM
===============

MMEditing can be installed via MIM, a dedicated package manage for OpenMMLab projects.
See `MIM documentations <https://openmim.readthedocs.io/en/latest/index.html>`_ for more details.


Another option: Docker Image
============================

We provide a `Dockerfile <https://github.com/open-mmlab/mmediting/blob/master/docker/Dockerfile>`_ for building a docker image.

To build the image:

```shell
# build an image with PyTorch 1.5, CUDA 10.1
docker build -t mmediting docker/
```

Run with:

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmediting/data mmediting
```

.. _Git: https://git-scm.com/
.. _Python: https://www.python.org/
.. _conda: https://docs.conda.io/en/latest/
.. _pip: https://pip.pypa.io/en/stable/
.. _pip: https://pip.pypa.io/en/stable/
.. _MMCV: https://github.com/open-mmlab/mmcv
.. _PyTorch: https://pytorch.org/
.. _CUDA driver version: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions
.. _end-of-life: https://endoflife.date/python
