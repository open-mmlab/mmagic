# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.testing import assert_allclose

from mmedit.models import GenDataPreprocessor


class TestBaseDataPreprocessor(TestCase):

    def test_init(self):
        data_preprocessor = GenDataPreprocessor(
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

    def test_forward(self):
        data_preprocessor = GenDataPreprocessor()
        input1 = torch.randn(3, 3, 5)
        input2 = torch.randn(3, 3, 5)
        label1 = torch.randn(1)
        label2 = torch.randn(1)

        # data = [
        #     dict(inputs=input1, data_sample=label1),
        #     dict(inputs=input2, data_sample=label2)
        # ]
        data = dict(
            inputs=torch.stack([input1, input2], dim=0),
            data_samples=[label1, label2])

        data = data_preprocessor(data)

        self.assertEqual(data['inputs']['img'].shape, (2, 3, 3, 5))

        target_input1 = (input1.clone() - 127.5) / 127.5
        target_input2 = (input2.clone() - 127.5) / 127.5
        assert_allclose(target_input1, data['inputs']['img'][0])
        assert_allclose(target_input2, data['inputs']['img'][1])
        assert_allclose(label1, data['data_samples'][0])
        assert_allclose(label2, data['data_samples'][1])

        # if torch.cuda.is_available():
        #     base_data_preprocessor = base_data_preprocessor.cuda()
        #     batch_inputs, batch_labels = base_data_preprocessor(data)
        #     self.assertEqual(batch_inputs.device.type, 'cuda')

        #     base_data_preprocessor = base_data_preprocessor.cpu()
        #     batch_inputs, batch_labels = base_data_preprocessor(data)
        #     self.assertEqual(batch_inputs.device.type, 'cpu')

        #     base_data_preprocessor = base_data_preprocessor.to('cuda:0')
        #     batch_inputs, batch_labels = base_data_preprocessor(data)
        #     self.assertEqual(batch_inputs.device.type, 'cuda')

        imgA1 = torch.randn(3, 3, 5)
        imgA2 = torch.randn(3, 3, 5)
        imgB1 = torch.randn(3, 3, 5)
        imgB2 = torch.randn(3, 3, 5)
        label1 = torch.randn(1)
        label2 = torch.randn(1)
        data = dict(
            inputs=dict(
                imgA=torch.stack([imgA1, imgA2], dim=0),
                imgB=torch.stack([imgB1, imgB2], dim=0)),
            data_samples=[label1, label2])
        data = data_preprocessor(data)
        self.assertEqual(list(data['inputs'].keys()), ['imgA', 'imgB'])

        img1 = torch.randn(3, 4, 4)
        img2 = torch.randn(3, 4, 4)
        noise1 = torch.randn(3, 4, 4)
        noise2 = torch.randn(3, 4, 4)
        target_input1 = (img1[[2, 1, 0], ...].clone() - 127.5) / 127.5
        target_input2 = (img2[[2, 1, 0], ...].clone() - 127.5) / 127.5

        data = dict(
            inputs=dict(
                noise=torch.stack([noise1, noise2], dim=0),
                img=torch.stack([img1, img2], dim=0),
                num_batches=[2, 2],
                mode=['ema', 'ema']))
        data_preprocessor = GenDataPreprocessor(rgb_to_bgr=True)
        # batch_inputs, batch_labels = data_preprocessor(data)
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

        # test dict input
        sampler_results = dict(inputs=dict(num_batches=2, mode='ema'))
        data = data_preprocessor(sampler_results)
        self.assertEqual(data['inputs'], sampler_results['inputs'])
        self.assertIsNone(data['data_samples'])
