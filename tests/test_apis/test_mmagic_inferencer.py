# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import pytest

from mmagic.apis import MMagicInferencer
from mmagic.utils import register_all_modules

register_all_modules()


def test_edit():
    with pytest.raises(Exception):
        MMagicInferencer('dog', ['error_type'], None)

    with pytest.raises(Exception):
        MMagicInferencer()

    with pytest.raises(Exception):
        MMagicInferencer(model_setting=1)

    supported_models = MMagicInferencer.get_inference_supported_models()
    MMagicInferencer.inference_supported_models_cfg_inited = False
    supported_models = MMagicInferencer.get_inference_supported_models()

    supported_tasks = MMagicInferencer.get_inference_supported_tasks()
    MMagicInferencer.inference_supported_models_cfg_inited = False
    supported_tasks = MMagicInferencer.get_inference_supported_tasks()

    task_supported_models = \
        MMagicInferencer.get_task_supported_models('Image2Image Translation')
    MMagicInferencer.inference_supported_models_cfg_inited = False
    task_supported_models = \
        MMagicInferencer.get_task_supported_models('Image2Image Translation')

    print(supported_models)
    print(supported_tasks)
    print(task_supported_models)

    cfg = osp.join(
        osp.dirname(__file__), '..', '..', 'configs', 'biggan',
        'biggan_2xb25-500kiters_cifar10-32x32.py')

    mmagic_instance = MMagicInferencer(
        'biggan',
        model_ckpt='',
        model_config=cfg,
        extra_parameters={'sample_model': 'ema'})
    mmagic_instance.print_extra_parameters()
    inference_result = mmagic_instance.infer(label=1)
    result_img = inference_result[1]
    assert result_img.shape == (4, 3, 32, 32)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
