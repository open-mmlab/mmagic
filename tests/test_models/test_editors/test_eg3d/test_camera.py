# Copyright (c) OpenMMLab. All rights reserved.
import math
from copy import deepcopy
from unittest import TestCase
from unittest.mock import patch

import torch
from mmengine.testing import assert_allclose

from mmagic.models.editors.eg3d.camera import (BaseCamera, GaussianCamera,
                                               UniformCamera)


class TestBaseCamera(TestCase):

    def setUp(self):
        self.default_cfg = dict(
            horizontal_mean=0,
            vertical_mean=3.141 / 2,
            horizontal_std=3.141 * 2,
            vertical_std=3.141 / 2,
            focal=1.025390625,
            radius=1.3)

    def test_init(self):
        cfg_ = deepcopy(self.default_cfg)
        camera = BaseCamera(**cfg_)
        self.assertEqual(camera.sampling_statregy.upper(), 'UNIFORM')

        cfg_ = deepcopy(self.default_cfg)
        cfg_['sampling_strategy'] = 'gaussian'
        camera = BaseCamera(**cfg_)
        self.assertEqual(camera.sampling_statregy.upper(), 'GAUSSIAN')

        # test FOV and FOCAL is both passed
        cfg_ = deepcopy(self.default_cfg)
        cfg_['fov'] = 2333
        with self.assertRaises(AssertionError):
            BaseCamera(**cfg_)

        # test FOV and FOCAL is neither passed
        cfg_ = deepcopy(self.default_cfg)
        cfg_['focal'] = None
        camera = BaseCamera(**cfg_)
        self.assertIsNone(camera.focal)
        self.assertIsNone(camera.fov)

    def test_repr(self):

        cfg_ = deepcopy(self.default_cfg)
        camera = BaseCamera(**cfg_)
        repr_string = repr(camera)
        attribute_list = [
            'horizontal_mean', 'vertical_mean', 'horizontal_std',
            'vertical_std', 'focal', 'look_at', 'up', 'radius',
            'sampling_statregy'
        ]
        for attr in attribute_list:
            self.assertIn(attr, repr_string)
        self.assertIn('BaseCamera', repr_string)

        cfg_ = deepcopy(self.default_cfg)
        cfg_['focal'] = None
        camera = BaseCamera(**cfg_)
        repr_string = repr(camera)
        self.assertNotIn('focal', repr_string)
        self.assertNotIn('fov', repr_string)

    def test_sample_intrinsic(self):
        target = [1.025390625, 0.0, 0.5, 0.0, 1.025390625, 0.5, 0.0, 0.0, 1.0]
        target = torch.FloatTensor(target).reshape(3, 3)

        # test sample with default focal
        cfg_ = deepcopy(self.default_cfg)
        camera = BaseCamera(**cfg_)
        intrinsic = camera.sample_intrinsic(batch_size=3)
        assert_allclose(intrinsic, target[None, ...].repeat(3, 1, 1))

        # test sample with default FOV
        cfg_ = deepcopy(self.default_cfg)
        cfg_['focal'] = None
        cfg_['fov'] = 69.18818449595669
        camera = BaseCamera(**cfg_)
        intrinsic = camera.sample_intrinsic(batch_size=4)
        assert_allclose(intrinsic, target[None, ...].repeat(4, 1, 1))

        # test sample fov and focal is passed at the same time
        with self.assertRaises(AssertionError):
            intrinsic = camera.sample_intrinsic(fov=1, focal=2, batch_size=4)

        # test focal is passed
        intrinsic = camera.sample_intrinsic(focal=1, batch_size=4)
        self.assertTrue((intrinsic[:, 0, 0] == 1).all())
        self.assertTrue((intrinsic[:, 1, 1] == 1).all())

        # test fov is passed
        target_focal = float(1 / (math.tan(1 * math.pi / 360) * 1.414))
        intrinsic = camera.sample_intrinsic(fov=1, batch_size=4)
        self.assertTrue((intrinsic[:, 0, 0] == target_focal).all())
        self.assertTrue((intrinsic[:, 1, 1] == target_focal).all())

        # test focal and fov is not passed and not be initialized
        cfg_ = deepcopy(self.default_cfg)
        cfg_['focal'] = None
        camera = BaseCamera(**cfg_)
        with self.assertRaises(ValueError):
            camera.sample_intrinsic(batch_size=4)

    def test_sample_camera2world(self):
        cfg_ = deepcopy(self.default_cfg)
        camera = BaseCamera(**cfg_)
        cam2world = camera.sample_camera2world()
        self.assertEqual(cam2world.shape, (1, 4, 4))

        mock_path = 'mmagic.models.editors.eg3d.camera.TORCH_VERSION'
        with patch(mock_path, '1.6.0'):
            print(torch.__version__)
            cfg_ = deepcopy(self.default_cfg)
            camera = BaseCamera(**cfg_)
            cam2world = camera.sample_camera2world()
            self.assertEqual(cam2world.shape, (1, 4, 4))

    def test_sample_in_range(self):
        cfg_ = deepcopy(self.default_cfg)
        cfg_['sampling_strategy'] = 'unknow'
        camera = BaseCamera(**cfg_)
        with self.assertRaises(ValueError):
            camera._sample_in_range(1, 1, 1)


class TestUniformCamera(TestCase):

    def setUp(self):
        self.default_cfg = dict(
            horizontal_mean=0,
            vertical_mean=3.141 / 2,
            horizontal_std=3.141 * 2,
            vertical_std=3.141 / 2,
            focal=1.025390625,
            radius=1.3)

    def test_init(self):
        cfg_ = deepcopy(self.default_cfg)
        camera = UniformCamera(**cfg_)
        self.assertEqual(camera.sampling_statregy.upper(), 'UNIFORM')

    def test_repr(self):
        cfg_ = deepcopy(self.default_cfg)
        camera = UniformCamera(**cfg_)
        repr_string = repr(camera)
        self.assertIn('UniformCamera', repr_string)


class TestGaussianCamera(TestCase):

    def setUp(self):
        self.default_cfg = dict(
            horizontal_mean=0,
            vertical_mean=3.141 / 2,
            horizontal_std=3.141 * 2,
            vertical_std=3.141 / 2,
            focal=1.025390625,
            radius=1.3)

    def test_init(self):
        cfg_ = deepcopy(self.default_cfg)
        camera = GaussianCamera(**cfg_)
        self.assertEqual(camera.sampling_statregy.upper(), 'GAUSSIAN')

    def test_repr(self):
        cfg_ = deepcopy(self.default_cfg)
        camera = GaussianCamera(**cfg_)
        repr_string = repr(camera)
        self.assertIn('GaussianCamera', repr_string)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
