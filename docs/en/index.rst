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
   :caption: Get Started

   1_overview.md
   2_get_started.md

.. toctree::
   :maxdepth: 1
   :caption: User Guides

   user_guides/1_config.md
   user_guides/2_dataset_prepare.md
   user_guides/3_inference.md
   user_guides/4_train_test.md
   user_guides/5_visualization.md
   user_guides/6_useful_tools.md
   user_guides/7_deploy.md
   user_guides/8_metrics.md

.. toctree::
   :maxdepth: 2
   :caption: Advanced Guides

   advanced_guides/1_models.md
   advanced_guides/2_dataset.md
   advanced_guides/3_transforms.md
   advanced_guides/4_losses.md

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
   :maxdepth: 1
   :caption: Model Zoo

   3_model_zoo.md


.. toctree::
   :maxdepth: 1
   :caption: Dataset Zoo

   dataset_zoo/0_overview.md
   dataset_zoo/1_super_resolution_datasets.md
   dataset_zoo/2_inpainting_datasets.md
   dataset_zoo/3_matting_datasets.md
   dataset_zoo/4_video_interpolation_datasets.md
   dataset_zoo/5_unconditional_gans_datasets.md
   dataset_zoo/6_image_translation_datasets.md


.. toctree::
   :maxdepth: 1
   :caption: Migration from MMEdit 0.x

   migration/1_overview.md
   migration/2_runtime.md
   migration/3_models.md
   migration/4_eval_test.md
   migration/5_schedule.md
   migration/6_data.md
   migration/7_distributed_train.md
   migration/8_optimizers.md
   migration/9_visualization.md
   migration/10_amp.md


.. toctree::
   :maxdepth: 1
   :caption: Notes

   notes/1_contribution_guide.md
   notes/2_projects.md
   notes/3_changelog.md
   notes/4_faq.md


.. toctree::
   :caption: Switch Language

   5_switch_language.md



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
