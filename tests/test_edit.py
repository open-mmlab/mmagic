# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from mmedit.edit import MMEdit
from mmedit.utils import register_all_modules

register_all_modules()


def test_edit():
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
