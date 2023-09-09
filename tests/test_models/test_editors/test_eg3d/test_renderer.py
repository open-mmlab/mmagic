# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from unittest import TestCase
from unittest.mock import Mock, patch

import torch

from mmagic.models.editors.eg3d.renderer import EG3DDecoder, EG3DRenderer


class TestEG3DRenderer(TestCase):

    def setUp(self) -> None:
        self.n_tri, self.tri_feat, self.tri_res = 3, 16, 8
        self.nerf_res = 10
        self.decoder_out_channels = 5
        self.decoder_rgb_padding = 0.233

        self.decoder_cfg = dict(
            in_channels=self.tri_feat,
            hidden_channels=10,
            out_channels=self.decoder_out_channels,
            rgb_padding=self.decoder_rgb_padding)
        self.renderer_cfg = dict(
            ray_start=0.1,
            ray_end=2.6,
            box_warp=1.6,
            depth_resolution=5,
            white_back=True,
            depth_resolution_importance=10,
            decoder_cfg=deepcopy(self.decoder_cfg))

    def test_init(self):
        cfg_ = deepcopy(self.renderer_cfg)
        renderer = EG3DRenderer(**cfg_)
        self.assertIsInstance(renderer.decoder, EG3DDecoder)
        self.assertEqual(renderer.decoder.rgb_padding,
                         self.decoder_rgb_padding)

    def test_forward(self):
        nerf_res = self.nerf_res
        n_tri, tri_feat, tri_res = self.n_tri, self.tri_feat, self.tri_res

        plane = torch.randn(4, n_tri, tri_feat, tri_res, tri_res)
        ray_origins = torch.randn(4, nerf_res * nerf_res, 3)
        ray_directions = torch.randn(4, nerf_res * nerf_res, 3)

        cfg_ = deepcopy(self.renderer_cfg)
        renderer = EG3DRenderer(**cfg_)
        rgb, depth, weights = renderer(plane, ray_origins, ray_directions)

        self.assertEqual(rgb.shape,
                         (4, nerf_res * nerf_res, self.decoder_out_channels))
        self.assertEqual(depth.shape, (4, nerf_res * nerf_res, 1))
        self.assertEqual(weights.shape, (4, nerf_res * nerf_res, 1))

        # test white_back is True + density_noise > 0
        cfg_ = deepcopy(self.renderer_cfg)
        cfg_['white_back'] = False
        cfg_['density_noise'] = 2
        renderer = EG3DRenderer(**cfg_)
        rgb, depth, weights = renderer(plane, ray_origins, ray_directions)

        self.assertEqual(rgb.shape,
                         (4, nerf_res * nerf_res, self.decoder_out_channels))
        self.assertEqual(depth.shape, (4, nerf_res * nerf_res, 1))
        self.assertEqual(weights.shape, (4, nerf_res * nerf_res, 1))

        # test ray_start and end is auto
        cfg_ = deepcopy(self.renderer_cfg)
        cfg_['ray_start'] = cfg_['ray_end'] = 'auto'
        renderer = EG3DRenderer(**cfg_)
        rgb, depth, weights = renderer(plane, ray_origins, ray_directions)

        self.assertEqual(rgb.shape,
                         (4, nerf_res * nerf_res, self.decoder_out_channels))
        self.assertEqual(depth.shape, (4, nerf_res * nerf_res, 1))
        self.assertEqual(weights.shape, (4, nerf_res * nerf_res, 1))

        # test clamp mode is valid
        cfg_ = deepcopy(self.renderer_cfg)
        cfg_['clamp_mode'] = 'unsupport'
        renderer = EG3DRenderer(**cfg_)
        with self.assertRaises(AssertionError):
            renderer(plane, ray_origins, ray_directions)

        # test without hierarchical sampling + test render_kwargs
        render_kwargs = dict(depth_resolution_importance=0)
        mock_func = Mock(return_value=(rgb, depth, weights[..., None]))
        mock_path = ('mmagic.models.editors.eg3d.renderer.EG3DRenderer.'
                     'volume_rendering')
        with patch(mock_path, new=mock_func):
            renderer = EG3DRenderer(**cfg_)
            renderer(
                plane,
                ray_origins,
                ray_directions,
                render_kwargs=render_kwargs)
            mock_func.assert_called_once()

        # cover TORCH_VERSION < 1.8.0
        mock_path = 'mmagic.models.editors.eg3d.renderer.TORCH_VERSION'
        with patch(mock_path, '1.6.0'):
            cfg_ = deepcopy(self.renderer_cfg)
            renderer = EG3DRenderer(**cfg_)
            renderer(
                plane,
                ray_origins,
                ray_directions,
                render_kwargs=render_kwargs)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
