# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.testing import assert_allclose

from mmedit.models.data_preprocessors import EditDataPreprocessor
from mmedit.structures import EditDataSample, PixelData


class TestBaseDataPreprocessor(TestCase):

    def test_init(self):
        data_preprocessor = EditDataPreprocessor(
            bgr_to_rgb=True,
            mean=[0, 0, 0],
            std=[255, 255, 255],
            pad_size_divisor=16,
            pad_value=10)

        self.assertEqual(data_preprocessor._device.type, 'cpu')
        self.assertTrue(data_preprocessor._channel_conversion, True)
        assert_allclose(data_preprocessor.mean,
                        torch.tensor([0, 0, 0]).view(-1, 1, 1))
        assert_allclose(data_preprocessor.std,
                        torch.tensor([255, 255, 255]).view(-1, 1, 1))
        assert_allclose(data_preprocessor.pad_value, torch.tensor(10))
        self.assertEqual(data_preprocessor.pad_size_divisor, 16)
        self.assertEqual(data_preprocessor.pad_mode, 'constant')
        self.assertFalse(data_preprocessor.only_norm_gt_in_training)

        # test non-image-keys
        data_preprocessor = EditDataPreprocessor(non_image_keys='feature')
        self.assertIn('feature', data_preprocessor._NON_IMAGE_KEYS)
        data_preprocessor = EditDataPreprocessor(non_image_keys=['feature'])
        self.assertIn('feature', data_preprocessor._NON_IMAGE_KEYS)

        # test non-concentate-keys
        data_preprocessor = EditDataPreprocessor(non_concentate_keys='n_imgs')
        self.assertIn('n_imgs', data_preprocessor._NON_CONCENTATE_KEYS)
        data_preprocessor = EditDataPreprocessor(
            non_concentate_keys=['n_imgs'])
        self.assertIn('n_imgs', data_preprocessor._NON_CONCENTATE_KEYS)

    def test_forward(self):
        data_preprocessor = EditDataPreprocessor()
        input1 = torch.randn(3, 3, 5)
        input2 = torch.randn(3, 3, 5)

        data = dict(inputs=[input1, input2])

        data = data_preprocessor(data)

        self.assertEqual(data['inputs']['img'].shape, (2, 3, 3, 5))

        target_input1 = (input1.clone() - 127.5) / 127.5
        target_input2 = (input2.clone() - 127.5) / 127.5
        assert_allclose(target_input1, data['inputs']['img'][0])
        assert_allclose(target_input2, data['inputs']['img'][1])

        imgA1 = torch.randn(3, 3, 5)
        imgA2 = torch.randn(3, 3, 5)
        imgB1 = torch.randn(3, 3, 5)
        imgB2 = torch.randn(3, 3, 5)
        data = dict(inputs=[
            dict(imgA=imgA1, imgB=imgB1),
            dict(imgA=imgA2, imgB=imgB2)
        ])
        data = data_preprocessor(data)
        self.assertEqual(list(data['inputs'].keys()), ['imgA', 'imgB'])

        img1 = torch.randn(3, 4, 4)
        img2 = torch.randn(3, 4, 4)
        noise1 = torch.randn(3, 4, 4)
        noise2 = torch.randn(3, 4, 4)
        target_input1 = (img1[[2, 1, 0], ...].clone() - 127.5) / 127.5
        target_input2 = (img2[[2, 1, 0], ...].clone() - 127.5) / 127.5

        data = dict(inputs=[
            dict(noise=noise1, img=img1, num_batches=2, mode='ema'),
            dict(noise=noise2, img=img2, num_batches=2, mode='ema'),
        ])
        data_preprocessor = EditDataPreprocessor(rgb_to_bgr=True)
        data = data_preprocessor(data)

        self.assertEqual(
            list(data['inputs'].keys()),
            ['noise', 'img', 'num_batches', 'mode'])
        assert_allclose(data['inputs']['noise'][0], noise1)
        assert_allclose(data['inputs']['noise'][1], noise2)
        assert_allclose(data['inputs']['img'][0], target_input1)
        assert_allclose(data['inputs']['img'][1], target_input2)
        self.assertEqual(data['inputs']['num_batches'], 2)
        self.assertEqual(data['inputs']['mode'], 'ema')

        # test tensor input
        data = dict(inputs=torch.stack([img1, img2], dim=0))
        data_preprocessor = EditDataPreprocessor(rgb_to_bgr=True)
        data = data_preprocessor(data)

        # test dict input
        sampler_results = dict(inputs=dict(num_batches=2, mode='ema'))
        data = data_preprocessor(sampler_results)
        self.assertEqual(data['inputs'], sampler_results['inputs'])
        self.assertIsNone(data['data_samples'])

        # test dict input with tensor
        sampler_results = dict(inputs=dict(fake_img=torch.randn(2, 3, 10, 10)))
        data = data_preprocessor(sampler_results)

        # test no-norm
        data_preprocessor = EditDataPreprocessor(mean=None, std=None)
        input1 = torch.randn(3, 3, 5)
        input2 = torch.randn(3, 3, 5)
        data = dict(inputs=torch.stack([input1, input2], dim=0))
        data = data_preprocessor(data)
        self.assertEqual(data['inputs']['img'].shape, (2, 3, 3, 5))
        self.assertTrue((data['inputs']['img'] == torch.stack([input1, input2],
                                                              dim=0)).all())

        # test do not norm GT data samples in test
        data_preprocessor = EditDataPreprocessor(only_norm_gt_in_training=True)
        gt_inp1 = torch.randint(0, 255, (3, 5, 5))
        gt_inp2 = torch.randint(0, 255, (3, 5, 5))

        data_samples = [
            EditDataSample(gt_img=PixelData(data=gt_inp1)),
            EditDataSample(gt_img=PixelData(data=gt_inp2))
        ]
        data = dict(inputs=[input1, input2], data_samples=data_samples)
        data = data_preprocessor(data, training=False)
        self.assertTrue((data['data_samples'][0].gt_img.data == gt_inp1).all())
        self.assertTrue((data['data_samples'][1].gt_img.data == gt_inp2).all())

        data = dict(inputs=[input1, input2], data_samples=data_samples)
        data = data_preprocessor(data, training=True)
        self.assertTrue((data['data_samples'][0].gt_img.data <= 1).all())
        self.assertTrue((data['data_samples'][1].gt_img.data <= 1).all())

        # test input with different shape
        input1 = torch.randn(3, 3, 5)
        input2 = torch.randn(3, 5, 3)
        data = dict(inputs=[input1, input2])
        data = data_preprocessor(data)
        self.assertEqual(data['inputs']['img'].shape, (2, 3, 5, 5))
        self.assertTrue(
            (data_preprocessor.pad_size_dict['img'] == torch.FloatTensor(
                [[0, 2, 0], [0, 0, 2]])).all())
        destruct_batch = data_preprocessor.destructor(data['inputs']['img'])
        # NOTE: this shape checking is aligned to the logic of current
        # version's `destruct` method
        self.assertEqual(destruct_batch.shape, (2, 3, 3, 5))

        # test output view is defined
        data_preprocessor.output_view = [1, -1]
        data_preprocessor.pad_size_dict['feat'] = None
        feat_input = torch.randn(256, 3)
        destruct_feat = data_preprocessor.destructor(feat_input, 'feat')
        self.assertEqual(destruct_feat.shape, (256, 3))

        # test destruct without norm
        data_preprocessor = EditDataPreprocessor(
            only_norm_gt_in_training=True, std=None, mean=None)
        input1 = torch.randint(0, 255, (1, 5, 5))
        input2 = torch.randint(0, 255, (1, 5, 5))
        data = dict(inputs=[input1, input2])
        data = data_preprocessor(data)
        destruct_batch = data_preprocessor.destructor(data['inputs']['img'])
        self.assertEqual(destruct_batch.shape, (2, 1, 5, 5))
        self.assertTrue((destruct_batch == torch.stack([input1, input2],
                                                       dim=0)).all())

        # test destruct pad_info is None
        data_preprocessor.pad_size_dict['feature'] = None
        feat_input = torch.randint(0, 255, (2, 3, 10, 10))
        destruct_feat = data_preprocessor.destructor(
            feat_input, target_key='feature')
        self.assertTrue((feat_input == destruct_feat).all())

        # test wrong input type
        data = dict(inputs='wrong type')
        with self.assertRaises(ValueError):
            data = data_preprocessor(data)

        # test skip norm data sample
        data_preprocessor = EditDataPreprocessor()
        inputs = [torch.randint(0, 255, (1, 5, 5))]
        data_samples = [EditDataSample()]
        data = data_preprocessor(
            dict(inputs=inputs, data_samples=data_samples))
        destruct_batch = data_preprocessor.destructor(data['inputs']['img'])
