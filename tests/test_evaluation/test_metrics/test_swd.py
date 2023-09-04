# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
import torch

from mmagic.evaluation import SlicedWassersteinDistance
from mmagic.models import DataPreprocessor
from mmagic.structures import DataSample


class TestSWD(TestCase):

    def test_init(self):
        swd = SlicedWassersteinDistance(fake_nums=10, image_shape=(3, 32, 32))
        self.assertEqual(len(swd.real_results), 2)

    def test_prosess(self):
        model = MagicMock()
        model.data_preprocessor = DataPreprocessor()
        swd = SlicedWassersteinDistance(fake_nums=100, image_shape=(3, 32, 32))
        swd.prepare(model, None)

        torch.random.manual_seed(42)
        real_samples = [
            dict(inputs=torch.rand(3, 32, 32) * 255.) for _ in range(100)
        ]
        fake_samples = [
            DataSample(
                fake_img=(torch.rand(3, 32, 32) * 255),
                gt_img=(torch.rand(3, 32, 32) * 255)).to_dict()
            for _ in range(100)
        ]

        swd.process(real_samples, fake_samples)
        # 100 samples are passed in 1 batch, _num_processed should be 100
        self.assertEqual(swd._num_processed, 100)
        # _num_processed(100) > fake_nums(4), _num_processed should be
        # unchanged
        swd.process(real_samples, fake_samples)
        self.assertEqual(swd._num_processed, 100)

        output = swd.evaluate()
        result = [16.495922580361366, 24.15413036942482, 20.325026474893093]
        output = [item / 100 for item in output.values()]
        result = [item / 100 for item in result]
        np.testing.assert_almost_equal(output, result, decimal=1)

        swd = SlicedWassersteinDistance(
            fake_nums=4,
            fake_key='fake',
            real_key='img',
            sample_model='orig',
            image_shape=(3, 32, 32))
        swd.prepare(model, None)

        # test gray scale input
        swd.image_shape = (1, 32, 32)
        real_samples = [
            dict(inputs=torch.rand(1, 32, 32) * 255.) for _ in range(100)
        ]
        fake_samples = [
            DataSample(
                fake_img=torch.rand(1, 32, 32) * 255,
                gt_img=torch.rand(1, 32, 32) * 255).to_dict()
            for _ in range(100)
        ]
        swd.process(real_samples, fake_samples)

        # test fake_nums is -1
        swd = SlicedWassersteinDistance(
            fake_nums=-1,
            fake_key='fake',
            real_key='img',
            sample_model='orig',
            image_shape=(3, 32, 32))
        fake_samples = [
            DataSample(
                fake_img=torch.rand(3, 32, 32) * 255,
                gt_img=torch.rand(3, 32, 32) * 255).to_dict()
            for _ in range(10)
        ]
        for _ in range(3):
            swd.process(None, fake_samples)
        # fake_nums is -1, all samples (10 * 3 = 30) is processed
        self.assertEqual(swd._num_processed, 30)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
