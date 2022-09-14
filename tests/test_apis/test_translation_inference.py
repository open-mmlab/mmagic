# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from mmengine import Config

from mmedit.apis import init_model, sample_img2img_model
from mmedit.utils import register_all_modules

register_all_modules()


def test_unconditional_inference():
    cfg = osp.join(
        osp.dirname(__file__), '..', '..', 'configs', 'pix2pix',
        'pix2pix_vanilla-unet-bn_1xb1-80kiters_facades.py')
    cfg = Config.fromfile(cfg)
    model = init_model(cfg, device='cpu')
    model.eval()
    data_path = osp.join(
        osp.dirname(__file__), '..', 'data', 'unpaired', 'trainA', '1.jpg')
    # test num_samples can be divided by num_batches
    results = sample_img2img_model(
        model, image_path=data_path, target_domain='photo')
    print(results.shape)
    assert results.shape == (1, 3, 256, 256)

    # test target domain is None
    results = sample_img2img_model(model, image_path=data_path)
    assert results.shape == (1, 3, 256, 256)
