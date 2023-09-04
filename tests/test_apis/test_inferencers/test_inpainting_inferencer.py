# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from mmagic.apis.inferencers.inpainting_inferencer import InpaintingInferencer
from mmagic.utils import register_all_modules

register_all_modules()


def test_inpainting_inferencer():
    data_root = osp.join(osp.dirname(__file__), '../../')
    masked_img_path = data_root + 'data/inpainting/celeba_test.png'
    mask_path = data_root + 'data/inpainting/bbox_mask.png'
    cfg = osp.join(
        osp.dirname(__file__),
        '..',
        '..',
        '..',
        'configs',
        'aot_gan',
        'aot-gan_smpgan_4xb4_places-512x512.py',
    )
    result_out_dir = osp.join(
        osp.dirname(__file__), '..', '..', 'data/out', 'inpainting_result.png')

    inferencer_instance = \
        InpaintingInferencer(cfg, None)
    inferencer_instance(img=masked_img_path, mask=mask_path)
    inference_result = inferencer_instance(
        img=masked_img_path, mask=mask_path, result_out_dir=result_out_dir)
    result_img = inference_result[1]
    assert result_img.shape == (256, 256, 3)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
