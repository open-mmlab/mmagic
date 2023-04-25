Welcome to MMagic's documentation!
=====================================

Languages:
`English <https://mmagic.readthedocs.io/en/latest/>`_
|
`简体中文 <https://mmagic.readthedocs.io/zh_CN/latest/>`_

MMagic (**M**\ultimodal **A**\dvanced, **G**\enerative, and **I**\ntelligent **C**\reation) is an open-source AIGC toolbox for professional AI researchers and machine learning engineers to explore image and video processing, editing and generation.

MMagic supports various foundamental generative models, including:

* Unconditional Generative Adversarial Networks (GANs)
* Conditional Generative Adversarial Networks (GANs)
* Internal Learning
* Diffusion Models
* And many other generative models are coming soon!

MMagic supports various applications, including:

- Text-to-Image
- Image-to-image translation
- 3D-aware generation
- Image super-resolution
- Video super-resolution
- Video frame interpolation
- Image inpainting
- Image matting
- Image restoration
- Image colorization
- Image generation
- And many other applications are coming soon!

MMagic is based on `PyTorch <https://pytorch.org>`_ and is a part of the `OpenMMLab project <https://openmmlab.com/>`_.
Codes are available on `GitHub <https://github.com/open-mmlab/mmagic>`_.


Documentation
=============

.. toctree::
   :maxdepth: 1
   :caption: Community

   community/contributing.md
   community/projects.md


.. toctree::
   :maxdepth: 1
   :caption: Get Started

   get_started/overview.md
   get_started/install.md
   get_started/quick_run.md


.. toctree::
   :maxdepth: 1
   :caption: User Guides

   user_guides/config.md
   user_guides/dataset_prepare.md
   user_guides/inference.md
   user_guides/train_test.md
   user_guides/metrics.md
   user_guides/visualization.md
   user_guides/useful_tools.md
   user_guides/deploy.md


.. toctree::
   :maxdepth: 2
   :caption: Advanced Guides

   advanced_guides/models.md
   advanced_guides/dataset.md
   advanced_guides/transforms.md
   advanced_guides/losses.md
   advanced_guides/evaluator.md
   advanced_guides/structures.md
   advanced_guides/data_preprocessor.md
   advanced_guides/data_flow.md


.. toctree::
   :maxdepth: 2
   :caption: How To

   howto/models.md
   howto/dataset.md
   howto/transforms.md
   howto/losses.md

.. toctree::
   :maxdepth: 1
   :caption: FAQ

   faq.md

.. toctree::
   :maxdepth: 2
   :caption: Model Zoo

   model_zoo/index.rst


.. toctree::
   :maxdepth: 1
   :caption: Dataset Zoo

   dataset_zoo/index.rst

.. toctree::
   :maxdepth: 1
   :caption: Changelog

   changelog.md

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   mmagic.apis.inferencers <autoapi/mmagic/apis/inferencers/index.rst>
   mmagic.structures <autoapi/mmagic/structures/index.rst>
   mmagic.datasets <autoapi/mmagic/datasets/index.rst>
   mmagic.datasets.transforms <autoapi/mmagic/datasets/transforms/index.rst>
   mmagic.evaluation <autoapi/mmagic/evaluation/index.rst>
   mmagic.visualization <autoapi/mmagic/visualization/index.rst>
   mmagic.engine.hooks <autoapi/mmagic/engine/hooks/index.rst>
   mmagic.engine.logging <autoapi/mmagic/engine/logging/index.rst>
   mmagic.engine.optimizers <autoapi/mmagic/engine/optimizers/index.rst>
   mmagic.engine.runner <autoapi/mmagic/engine/runner/index.rst>
   mmagic.engine.schedulers <autoapi/mmagic/engine/schedulers/index.rst>
   mmagic.models.archs <autoapi/mmagic/models/archs/index.rst>
   mmagic.models.base_models <autoapi/mmagic/models/base_models/index.rst>
   mmagic.models.losses <autoapi/mmagic/models/losses/index.rst>
   mmagic.models.data_preprocessors <autoapi/mmagic/models/data_preprocessors/index.rst>
   mmagic.models.utils <autoapi/mmagic/models/losses/utils.rst>
   mmagic.models.editors <autoapi/mmagic/models/editors/index.rst>
   mmagic.utils <autoapi/mmagic/utils/index.rst>


.. toctree::
   :maxdepth: 1
   :caption: Migration from MMEdit 0.x

   migration/overview.md
   migration/runtime.md
   migration/models.md
   migration/eval_test.md
   migration/schedule.md
   migration/data.md
   migration/distributed_train.md
   migration/optimizers.md
   migration/visualization.md
   migration/amp.md


.. toctree::
   :maxdepth: 1
   :caption: Device Support

   device/npu.md


.. toctree::
   :caption: Switch Language

   switch_language.md



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
