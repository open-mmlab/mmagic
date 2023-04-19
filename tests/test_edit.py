# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import pytest

from mmagic.edit import MMEdit
from mmagic.utils import register_all_modules

register_all_modules()


def test_edit():
    with pytest.raises(Exception):
        MMEdit('dog', ['error_type'], None)

    with pytest.raises(Exception):
        MMEdit()

    with pytest.raises(Exception):
        MMEdit(model_setting=1)

    supported_models = MMEdit.get_inference_supported_models()
    MMEdit.inference_supported_models_cfg_inited = False
    supported_models = MMEdit.get_inference_supported_models()

    supported_tasks = MMEdit.get_inference_supported_tasks()
    MMEdit.inference_supported_models_cfg_inited = False
    supported_tasks = MMEdit.get_inference_supported_tasks()

    task_supported_models = \
        MMEdit.get_task_supported_models('Image2Image Translation')
    MMEdit.inference_supported_models_cfg_inited = False
    task_supported_models = \
        MMEdit.get_task_supported_models('Image2Image Translation')

    print(supported_models)
    print(supported_tasks)
    print(task_supported_models)

    cfg = osp.join(
        osp.dirname(__file__), '..', 'configs', 'biggan',
        'biggan_2xb25-500kiters_cifar10-32x32.py')

    mmedit_instance = MMEdit(
        'biggan',
        model_ckpt='',
        model_config=cfg,
        extra_parameters={'sample_model': 'ema'})
    mmedit_instance.print_extra_parameters()
    inference_result = mmedit_instance.infer(label=1)
    result_img = inference_result[1]
    assert result_img.shape == (4, 3, 32, 32)


if __name__ == '__main__':
    test_edit()
