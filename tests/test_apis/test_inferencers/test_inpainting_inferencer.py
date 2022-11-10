# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from mmedit.apis.inferencers.inpainting_inferencer import InpaintingInferencer
from mmedit.utils import register_all_modules

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
    inferencer_instance = \
        InpaintingInferencer(cfg, None)
    inference_result = inferencer_instance(img=masked_img_path, mask=mask_path)
    result_img = inference_result[1]
    assert result_img.detach().cpu().numpy().shape == (3, 256, 256)


if __name__ == '__main__':
    test_inpainting_inferencer()
