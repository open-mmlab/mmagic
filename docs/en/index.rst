Welcome to MMEditing's documentation!
=====================================

Languages:
`English <https://mmediting.readthedocs.io/en/latest/>`_
|
`简体中文 <https://mmediting.readthedocs.io/zh_CN/latest/>`_

MMEditing is an open-source toolbox for image and video processing, editing and synthesis.

MMEditing supports various foundamental generative models, including:

* Unconditional Generative Adversarial Networks (GANs)
* Conditional Generative Adversarial Networks (GANs)
* Internal Learning
* Diffusion Models
* And many other generative models are coming soon!


MMEditing supports various applications, including:

* Image super-resolution
* Video super-resolution
* Video frame interpolation
* Image inpainting
* Image matting
* Image-to-image translation
* And many other applications are coming soon!

MMEditing is based on `PyTorch <https://pytorch.org>`_ and is a part of the `OpenMMLab project <https://openmmlab.com/>`_.
Codes are available on `GitHub <https://github.com/open-mmlab/mmediting>`_.


Documentation
=============

.. toctree::
   :maxdepth: 1
   :caption: Community

   community/contribution_guide.md
   community/projects.md
   community/changelog.md


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
   user_guides/faq.md


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
   :maxdepth: 2
   :caption: API Reference

   mmedit.apis.inferencers <autoapi/mmedit/apis/inferencers/index.rst>
   mmedit.structures <autoapi/mmedit/structures/index.rst>
   mmedit.datasets <autoapi/mmedit/datasets/index.rst>
   mmedit.datasets.transforms <autoapi/mmedit/datasets/transforms/index.rst>
   mmedit.evaluation <autoapi/mmedit/evaluation/index.rst>
   mmedit.visualization <autoapi/mmedit/visualization/index.rst>
   mmedit.engine.hooks <autoapi/mmedit/engine/hooks/index.rst>
   mmedit.engine.logging <autoapi/mmedit/engine/logging/index.rst>
   mmedit.engine.optimizers <autoapi/mmedit/engine/optimizers/index.rst>
   mmedit.engine.runner <autoapi/mmedit/engine/runner/index.rst>
   mmedit.engine.schedulers <autoapi/mmedit/engine/schedulers/index.rst>
   mmedit.models.base_archs <autoapi/mmedit/models/base_archs/index.rst>
   mmedit.models.base_models <autoapi/mmedit/models/base_models/index.rst>
   mmedit.models.losses <autoapi/mmedit/models/losses/index.rst>
   mmedit.models.data_preprocessors <autoapi/mmedit/models/data_preprocessors/index.rst>
   mmedit.models.utils <autoapi/mmedit/models/losses/utils.rst>
   mmedit.models.editors <autoapi/mmedit/models/editors/index.rst>
   mmedit.utils <autoapi/mmedit/utils/index.rst>


.. toctree::
   :maxdepth: 2
   :caption: Model Zoo

   model_zoo/index.rst


.. toctree::
   :maxdepth: 1
   :caption: Dataset Zoo

   dataset_zoo/overview.md
   dataset_zoo/super_resolution_datasets.md
   dataset_zoo/inpainting_datasets.md
   dataset_zoo/matting_datasets.md
   dataset_zoo/video_interpolation_datasets.md
   dataset_zoo/unconditional_gans_datasets.md
   dataset_zoo/image_translation_datasets.md


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
   :caption: Switch Language

   switch_language.md



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
