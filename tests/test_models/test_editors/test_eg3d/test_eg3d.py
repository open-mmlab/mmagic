# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from unittest import TestCase
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from mmagic.models.editors.eg3d.eg3d import EG3D
from mmagic.structures import DataSample
from mmagic.utils import register_all_modules

register_all_modules()


class TestEG3D(TestCase):

    def setUp(self):
        self.generator_cfg = dict(
            type='TriplaneGenerator',
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
        self.camera_cfg = dict(
            type='UniformCamera',
            horizontal_mean=3.141,
            horizontal_std=3.141,
            vertical_mean=3.141 / 2,
            vertical_std=3.141 / 2,
            focal=1.025390625,
            up=[0, 0, 1],
            radius=1.2)
        # self.discriminator_cfg = dict()
        self.default_cfg = dict(
            generator=self.generator_cfg,
            camera=self.camera_cfg,
            data_preprocessor=dict(type='DataPreprocessor'))

    def test_init(self):
        cfg_ = deepcopy(self.default_cfg)
        model = EG3D(**cfg_)
        self.assertEqual(model.noise_size, 8)
        self.assertEqual(model.num_classes, None)
        self.assertIsNotNone(model.camera)

        # test camera is None
        cfg_ = deepcopy(self.default_cfg)
        cfg_['camera'] = None
        model = EG3D(**cfg_)
        self.assertIsNone(model.camera)

        # test camers is module
        camera_mock = MagicMock(spec=nn.Module)
        cfg_['camera'] = camera_mock
        model = EG3D(**cfg_)
        self.assertEqual(model.camera, camera_mock)

    def test_label_fn(self):
        # test camera is None
        cfg_ = deepcopy(self.default_cfg)
        cfg_['camera'] = None
        model = EG3D(**cfg_)
        label = torch.randn(1, 25)
        self.assertTrue((label == model.label_fn(label)).all())

        with self.assertRaises(AssertionError):
            model.label_fn(None)

    def _check_datasample_output(self, outputs, out_size, n_points):
        target_keys = [
            'fake_img', 'depth', 'lr_img', 'ray_origins', 'ray_directions'
        ]
        target_shape = [(3, out_size, out_size), (1, n_points, n_points),
                        (3, n_points, n_points), (n_points**2, 3),
                        (n_points**2, 3)]
        for output in outputs:
            for key, shape in zip(target_keys, target_shape):
                self.assertTrue(hasattr(output, key))
                self.assertEqual(getattr(output, key).shape, shape)

    def _check_dict_output(self, outputs, out_size, n_points, bz):
        target_keys = [
            'fake_img', 'depth', 'lr_img', 'ray_origins', 'ray_directions'
        ]
        target_shape = [(bz, 3, out_size, out_size),
                        (bz, 1, n_points, n_points),
                        (bz, 3, n_points, n_points), (bz, n_points**2, 3),
                        (bz, n_points**2, 3)]
        for output in outputs:
            for key, shape in zip(target_keys, target_shape):
                self.assertIn(key, output)
                self.assertEqual(output[key].shape, shape)

    def test_forward(self):
        cfg_ = deepcopy(self.default_cfg)
        model = EG3D(**cfg_)
        outputs = model(dict(num_batches=4))
        self.assertEqual(len(outputs), 4)

        # test label is passed
        data_samples = []
        for _ in range(4):
            data_sample = DataSample()
            data_sample.set_gt_label(torch.randn(25))
            data_samples.append(data_sample)
        data_samples = DataSample.stack(data_samples)
        outputs = model(dict(num_batches=4), data_samples)
        self.assertEqual(len(outputs), 4)
        self._check_datasample_output(outputs, 32, 5)

        # test noise is passed
        outputs = model(torch.randn(4, 8))
        self.assertEqual(len(outputs), 4)
        self._check_datasample_output(outputs, 32, 5)

        # test orig/ema
        cfg_ = deepcopy(self.default_cfg)
        cfg_['ema_config'] = dict(interval=1)
        model = EG3D(**cfg_)
        outputs = model(dict(num_batches=4, sample_model='ema/orig'))
        self.assertEqual(len(outputs), 4)
        self.assertTrue(all([hasattr(output, 'orig') for output in outputs]))
        self.assertTrue(all([hasattr(output, 'ema') for output in outputs]))
        self._check_datasample_output([output.orig for output in outputs], 32,
                                      5)
        self._check_datasample_output([output.ema for output in outputs], 32,
                                      5)

    def test_interpolation(self):
        cfg_ = deepcopy(self.default_cfg)
        model = EG3D(**cfg_)
        output_list = model.interpolation(num_images=3, num_batches=2)
        self.assertEqual(len(output_list), 3)
        self._check_dict_output(output_list, 32, 5, 2)

        output_list = model.interpolation(
            num_images=3, num_batches=2, mode='camera', show_pbar=False)
        self.assertEqual(len(output_list), 3)
        self._check_dict_output(output_list, 32, 5, 2)

        output_list = model.interpolation(
            num_images=3, num_batches=2, mode='conditioning', show_pbar=False)
        self.assertEqual(len(output_list), 3)
        self._check_dict_output(output_list, 32, 5, 2)

        # test ema
        cfg_ = deepcopy(self.default_cfg)
        cfg_['ema_config'] = dict(interval=1)
        model = EG3D(**cfg_)

        output_list = model.interpolation(
            num_images=3, num_batches=2, sample_model='ema')
        self.assertEqual(len(output_list), 3)
        self._check_dict_output(output_list, 32, 5, 2)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
