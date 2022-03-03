Installation
############

MMEditing is a Python toolbox based on `PyTorch`_ and `MMCV`_.
These packages can be installed in your system in various ways.
Here we describe some best practices.

If you have difficulties in installing MMEditing, please feel free to `open a discussion <https://github.com/open-mmlab/mmediting/discussions>`_ to describe your situations.


Prerequisites
===================

* `Python`_ >= 3.6
* `pip`_ and/or `conda`_
* `Git`_
* (Recommended) NVIDIA GPU with driver version >= 440.33 (Linux) or >= 441.22 (Windows)
* (Optional) CUDA and C++ (GCC / Clang / MSVC) compilers if you need to compile `MMCV`_ from source codes

.. warning::

   As Python 3.6 has reached end-of-life on 23-Dec-2021, we will drop its support in the future.


Install CPU Version
===================

MMEditing is fully supported on CPUs, despite the slow running speed.
However, the CPU version is much more lightweight.
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

or

.. code-block:: sh

   pip install torch torchvision

See `PyTorch installation guide <https://pytorch.org/get-started/locally/>`_ for more installation options.


Step 3.
Install pre-compiled MMCV for CPU environment

.. code-block:: sh

   pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.10/index.html

See `MMCV documentations <https://mmcv.readthedocs.io/en/latest/get_started/installation.html>`_ for more installation options.


.. note::

   The download link of MMCV depends on the version of PyTorch.
   The above link is for **PyTorch 1.10.x**.
   If you use a different version of PyTorch, you need to change the link accordingly.

   E.g. ``https://download.openmmlab.com/mmcv/dist/cpu/`` **torch1.9** ``/index.html`` for **PyTorch 1.9.x**.


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

Install CUDA Version
====================

To enable the full power of MMEditing, we recommend installing the GPU version.
The only difference to CPU version lies at the installation of PyTorch and MMCV.
Please pay more attention to the version and CUDA version of PyTorch and MMCV.


Step 1.
Create and activate a conda virtual environment

.. code-block:: sh

   conda create --name mmedit python=3.8 -y
   conda activate mmedit


Step 2.
Install the GPU version of PyTorch and torchvision

.. code-block:: sh

   conda install pytorch torchvision cpuonly -c pytorch

or

.. code-block:: sh

   pip install torch torchvision

See `PyTorch installation guide <https://pytorch.org/get-started/locally/>`_ for more installation options.


Step 3.
Install pre-compiled MMCV for CPU environment

.. code-block:: sh

   pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.10/index.html

See `MMCV documentations <https://mmcv.readthedocs.io/en/latest/get_started/installation.html>`_ for more installation options.


.. note::

   The download link of MMCV depends on the version of PyTorch.
   The above link is for **PyTorch 1.10.x**.
   If you use a different version of PyTorch, you need to change the link accordingly.

   E.g. ``https://download.openmmlab.com/mmcv/dist/cpu/`` **torch1.9** ``/index.html`` for **PyTorch 1.9.x**.


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

MMEditing can also be installed via MIM.
See `MIM documentations <https://openmim.readthedocs.io/en/latest/index.html>`_ for more details.




.. _Git: https://git-scm.com/
.. _Python: https://www.python.org/
.. _conda: https://docs.conda.io/en/latest/
.. _pip: https://pip.pypa.io/en/stable/
.. _pip: https://pip.pypa.io/en/stable/
.. _MMCV: https://github.com/open-mmlab/mmcv
.. _PyTorch: https://pytorch.org/
.. _CUDA version table: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions
.. _end-of-life: https://endoflife.date/python
