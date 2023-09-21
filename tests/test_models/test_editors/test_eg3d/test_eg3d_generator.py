# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from unittest import TestCase
from unittest.mock import Mock, patch

import torch

from mmagic.models.editors.eg3d.eg3d_generator import TriplaneGenerator


class TestEG3DGenerator(TestCase):

    def setUp(self):
        self.default_cfg = dict(
            out_size=32,
            noise_size=8,
            style_channels=8,
            num_mlps=1,
            triplane_size=8,
            triplane_channels=4,
            sr_in_size=8,
            sr_in_channels=8,
            neural_rendering_resolution=5,
            cond_scale=1,
            renderer_cfg=dict(
                ray_start=0.1,
                ray_end=2.6,
                box_warp=1.6,
                depth_resolution=4,
                white_back=True,
                depth_resolution_importance=4,
            ),
            rgb2bgr=True)

    def test_init(self):
        cfg_ = deepcopy(self.default_cfg)
        gen = TriplaneGenerator(**cfg_)

        # check decoder in/out channels
        decoder_in_chns = gen.renderer.decoder.net[0].weight.shape[1]
        decoder_out_chns = gen.renderer.decoder.net[-1].weight.shape[0]
        self.assertEqual(decoder_in_chns, 4)
        self.assertEqual(decoder_out_chns, 8 + 1)  # (sr_in_channels + 1)

        # test sr_factor not in [2, 4, 8]
        cfg_ = deepcopy(self.default_cfg)
        cfg_['out_size'] = 64
        with self.assertRaises(AssertionError):
            TriplaneGenerator(**cfg_)

        cfg_['out_size'] = 10
        with self.assertRaises(AssertionError):
            TriplaneGenerator(**cfg_)

    def test_forward(self):
        cfg_ = deepcopy(self.default_cfg)
        gen = TriplaneGenerator(**cfg_)
        noise = torch.randn(2, 8)
        cond = torch.randn(2, 25)
        out = gen(noise, cond)
        self.assertEqual(out['fake_img'].shape, (2, 3, 32, 32))
        self.assertEqual(out['lr_img'].shape, (2, 3, 5, 5))
        self.assertEqual(out['depth'].shape, (2, 1, 5, 5))
        self.assertEqual(out['ray_directions'].shape, (2, 25, 3))
        self.assertEqual(out['ray_origins'].shape, (2, 25, 3))

        # test render_kwargs is work in forward
        render_kwargs = dict(a=1, b='b')
        render_mock = Mock(
            return_value=(torch.randn(2, 25, 8), torch.randn(2, 25, 1), None))
        patch_func = 'mmagic.models.editors.eg3d.renderer.EG3DRenderer.forward'
        with patch(patch_func, new=render_mock):
            gen = TriplaneGenerator(**cfg_)
            gen(noise, cond, render_kwargs=render_kwargs)
            _, called_kwargs = render_mock.call_args
            self.assertIn('render_kwargs', called_kwargs)
            self.assertDictEqual(called_kwargs['render_kwargs'], render_kwargs)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
