# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import shutil
from unittest import TestCase

import torch

from mmagic.structures import DataSample
from mmagic.utils import register_all_modules
from mmagic.visualization import Visualizer

register_all_modules()


class TestGenVisualer(TestCase):

    def test_add_datasample(self):
        visualizer = Visualizer(
            save_dir='tmp_dir', vis_backends=[dict(type='VisBackend')])

        img_root = osp.join('tmp_dir', 'vis_data', 'vis_image')
        # construct gen sample to vis
        gen_sample = [
            DataSample(
                fake_img=torch.randn(3, 16, 16),
                gt_img=torch.randn(1, 16, 16),
                ema=DataSample(fake_img=torch.randn(10, 3, 16, 16)),
                gif=torch.randn(10, 3, 16, 16),
                something=DataSample(
                    new=DataSample(img=torch.randn(3, 16, 16))))
            for _ in range(3)
        ]
        visualizer.add_datasample(
            name='fake_img',
            gen_samples=gen_sample,
            target_mean=None,
            target_std=None,
            target_keys=['fake_img', 'gt_img'],
            step=0)
        osp.exists(osp.join(img_root, 'fake_img_0.png'))

        visualizer.add_datasample(
            name='target_is_none',
            gen_samples=gen_sample,
            target_mean=None,
            target_std=None,
            step=1)
        osp.exists(osp.join(img_root, 'target_is_none_1.png'))

        visualizer.add_datasample(
            name='target_is_none_gif',
            gen_samples=gen_sample,
            target_mean=None,
            target_std=None,
            vis_mode='gif',
            step=2)
        osp.exists(osp.join(img_root, 'target_is_none_gif_2.gif'))

        visualizer.add_datasample(
            name='something',
            gen_samples=gen_sample,
            n_row=3,
            target_keys='something.new',
            step=3)
        osp.exists(osp.join(img_root, 'something_3.png'))

        visualizer.add_datasample(
            name='ema_padding',
            gen_samples=gen_sample,
            n_row=2,
            color_order='rgb',
            target_keys='ema',
            vis_mode='gif',
            step=4)
        osp.exists(osp.join(img_root, 'emd_padding_4.gif'))

        visualizer.add_datasample(
            name='fake_img_padding',
            gen_samples=gen_sample,
            target_mean=None,
            target_std=None,
            target_keys=['fake_img', 'gt_img'],
            n_row=4,
            step=5)
        osp.exists(osp.join(img_root, 'fake_img_padding_5.png'))

        shutil.rmtree(img_root)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
