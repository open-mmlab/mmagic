# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
import torch.nn.functional as F
from mmengine.testing import assert_allclose

from mmagic.models.data_preprocessors import DataPreprocessor
from mmagic.structures import DataSample


class TestBaseDataPreprocessor(TestCase):

    def test_init(self):
        data_preprocessor = DataPreprocessor(
            mean=[0, 0, 0],
            std=[255, 255, 255],
            pad_size_divisor=16,
            pad_value=10)

        self.assertEqual(data_preprocessor._device.type, 'cpu')
        assert_allclose(data_preprocessor.mean,
                        torch.tensor([0, 0, 0]).view(-1, 1, 1))
        assert_allclose(data_preprocessor.std,
                        torch.tensor([255, 255, 255]).view(-1, 1, 1))
        assert_allclose(data_preprocessor.pad_value, torch.tensor(10))
        self.assertEqual(data_preprocessor.pad_size_divisor, 16)
        self.assertEqual(data_preprocessor.pad_mode, 'constant')

        # test non-image-keys
        data_preprocessor = DataPreprocessor(non_image_keys='feature')
        self.assertIn('feature', data_preprocessor._NON_IMAGE_KEYS)
        data_preprocessor = DataPreprocessor(non_image_keys=['feature'])
        self.assertIn('feature', data_preprocessor._NON_IMAGE_KEYS)

        # test non-concentate-keys
        data_preprocessor = DataPreprocessor(non_concentate_keys='n_imgs')
        self.assertIn('n_imgs', data_preprocessor._NON_CONCATENATE_KEYS)
        data_preprocessor = DataPreprocessor(non_concentate_keys=['n_imgs'])
        self.assertIn('n_imgs', data_preprocessor._NON_CONCATENATE_KEYS)

    def test_parse_channel_index(self):
        data_preprocessor = DataPreprocessor()
        self.assertEqual(
            data_preprocessor._parse_channel_index(torch.rand(3, 5, 5)), 0)
        self.assertEqual(
            data_preprocessor._parse_channel_index(torch.rand(2, 3, 5, 5)), 1)
        self.assertEqual(
            data_preprocessor._parse_channel_index(torch.rand(2, 10, 3, 5, 5)),
            2)
        self.assertEqual(
            data_preprocessor._parse_channel_index(torch.rand(5, 5)), 1)
        with self.assertRaises(AssertionError):
            data_preprocessor._parse_channel_index(
                torch.rand(1, 2, 10, 3, 5, 5, 5))

        # test dict input
        inputs = dict(fake_img=torch.rand(2, 3, 5, 5))
        self.assertEqual(data_preprocessor._parse_channel_index(inputs), 1)

    def test_parse_channel_order(self):
        data_preprocessor = DataPreprocessor()
        parse_fn = data_preprocessor._parse_channel_order
        # test no data sample
        self.assertEqual(parse_fn('img', torch.rand(3, 5, 5)), 'BGR')
        self.assertEqual(parse_fn('img', torch.rand(1, 3, 3, 4)), 'BGR')
        self.assertEqual(parse_fn('img', torch.rand(1, 1, 3, 4)), 'single')

        # test dict input
        inputs = dict(fake_img=torch.rand(1, 3, 3, 4))
        self.assertEqual(parse_fn('img', inputs), 'BGR')

        # test data sample is not None
        data_sample = DataSample()

        # test gt_img key
        # - RGB in channel order --> RGB
        data_sample.set_metainfo({'gt_channel_order': 'RGB'})
        self.assertEqual(
            parse_fn('gt_img', torch.rand(3, 5, 5), data_sample), 'RGB')
        # - BGR in channel order --> BGR
        data_sample.set_metainfo({'gt_channel_order': 'BGR'})
        self.assertEqual(
            parse_fn('gt_img', torch.rand(3, 5, 5), data_sample), 'BGR')

        # - grayscale in color_type --> single
        data_sample.set_metainfo({'gt_color_type': 'grayscale'})
        self.assertEqual(
            parse_fn('gt_img', torch.rand(1, 5, 5), data_sample), 'single')
        with self.assertRaises(AssertionError):
            parse_fn('gt_img', torch.rand(3, 5, 5), data_sample)

        # - unchanged in color_type --> BGR / single
        data_sample.set_metainfo({'gt_color_type': 'unchanged'})
        self.assertEqual(
            parse_fn('gt_img', torch.rand(1, 5, 5), data_sample), 'single')
        self.assertEqual(
            parse_fn('gt_img', torch.rand(3, 5, 5), data_sample), 'BGR')

        # test non-gt keys
        # - RGB in channel order --> RGB
        data_sample.set_metainfo({'AAA_channel_order': 'RGB'})
        self.assertEqual(
            parse_fn('AAA', torch.rand(3, 5, 5), data_sample), 'RGB')
        # - BGR in channel order --> BGR
        data_sample.set_metainfo({'BBC_channel_order': 'BGR'})
        self.assertEqual(
            parse_fn('BBC', torch.rand(3, 5, 5), data_sample), 'BGR')

        # - grayscale in color_type --> single
        data_sample.set_metainfo({'mm_img_color_type': 'grayscale'})
        data_sample.set_metainfo({'mm_img_channel_order': 'RGB'})
        self.assertEqual(
            parse_fn('mm_img', torch.rand(1, 5, 5), data_sample), 'single')
        with self.assertRaises(AssertionError):
            parse_fn('mm_img', torch.rand(3, 5, 5), data_sample)

        # - unchanged in color_type --> BGR / single
        data_sample.set_metainfo({'mm_img_color_type': 'unchanged'})
        self.assertEqual(
            parse_fn('mm_img', torch.rand(1, 5, 5), data_sample), 'single')
        self.assertEqual(
            parse_fn('mm_img', torch.rand(3, 5, 5), data_sample), 'BGR')

        # - color_type and channel_order both None --> BGR / single
        self.assertEqual(
            parse_fn('dk', torch.rand(1, 5, 5), data_sample), 'single')
        self.assertEqual(
            parse_fn('dk', torch.rand(3, 5, 5), data_sample), 'BGR')

        # test parse channel order for a stacked data sample
        stacked_data_sample = DataSample.stack([data_sample, data_sample])
        self.assertEqual(
            parse_fn('mm_img', torch.rand(1, 5, 5), stacked_data_sample),
            'single')
        self.assertEqual(
            parse_fn('AAA', torch.rand(3, 5, 5), data_sample), 'RGB')

    def test_parse_batch_channel_order(self):
        data_preprocessor = DataPreprocessor()
        parse_fn = data_preprocessor._parse_batch_channel_order

        with self.assertRaises(AssertionError):
            parse_fn('img', torch.rand(1, 3, 5, 5), [DataSample()] * 2)

        inputs_batch = torch.randn(2, 3, 5, 5)
        data_sample_batch = [
            DataSample(metainfo=dict(gt_channel_order='RGB')),
            DataSample(metainfo=dict(gt_channel_order='RGB'))
        ]
        self.assertTrue(parse_fn('gt', inputs_batch, data_sample_batch))

        data_sample_batch = [
            DataSample(metainfo=dict(mm_img_channel_order='RGB')),
            DataSample(metainfo=dict(mm_img_channel_order='BGR'))
        ]
        with self.assertRaises(AssertionError):
            parse_fn(parse_fn('mm_img', inputs_batch, data_sample_batch))

    def test_do_conversion(self):
        data_preprocessor = DataPreprocessor()
        cov_fn = data_preprocessor._do_conversion

        # RGB -> BGR
        inputs = torch.rand(2, 3, 5, 5)
        target_outputs = inputs[:, [2, 1, 0], ...]
        outputs, order = cov_fn(inputs, 'RGB', 'BGR')
        self.assertTrue((outputs == target_outputs).all())
        self.assertEqual(order, 'BGR')

        # BGR -> RGB
        outputs, order = cov_fn(inputs, 'BGR', 'RGB')
        self.assertTrue((outputs == target_outputs).all())
        self.assertEqual(order, 'RGB')

        # RGB -> None
        target_outputs = inputs.clone()
        outputs, order = cov_fn(inputs, 'RGB', None)
        self.assertTrue((outputs == target_outputs).all())
        self.assertEqual(order, 'RGB')

        # BGR -> None
        target_outputs = inputs.clone()
        outputs, order = cov_fn(inputs, 'BGR', None)
        self.assertTrue((outputs == target_outputs).all())
        self.assertEqual(order, 'BGR')

        # single inputs -> None
        inputs = torch.rand(1, 10, 1, 5, 5)
        target_outputs = inputs.clone()
        outputs, order = cov_fn(inputs, 'single', None)
        self.assertTrue((outputs == target_outputs).all())
        self.assertEqual(order, 'single')

        # single inputs -> BGR / RGB
        outputs, order = cov_fn(inputs, 'single', None)
        self.assertTrue((outputs == target_outputs).all())
        self.assertEqual(order, 'single')
        outputs, order = cov_fn(inputs, 'single', 'RGB')
        self.assertTrue((outputs == target_outputs).all())
        self.assertEqual(order, 'single')
        outputs, order = cov_fn(inputs, 'single', 'BGR')
        self.assertTrue((outputs == target_outputs).all())
        self.assertEqual(order, 'single')

        with self.assertRaises(ValueError):
            cov_fn(inputs, 'RGBA', 'RGB')

        # RGBA inputs -> BGR / RGB
        inputs = torch.rand(2, 4, 5, 5)
        target_outputs = inputs.clone()
        outputs, order = cov_fn(inputs, 'RGB', 'RGB')
        self.assertTrue((outputs == target_outputs).all())
        self.assertEqual(order, 'RGB')
        target_outputs = inputs[:, [2, 1, 0, 3], ...]
        outputs, order = cov_fn(inputs, 'RGB', 'BGR')
        self.assertTrue((outputs == target_outputs).all())
        self.assertEqual(order, 'BGR')

    def test_update_metainfo(self):
        data_preprocessor = DataPreprocessor()
        update_fn = data_preprocessor._update_metainfo

        with self.assertRaises(AssertionError):
            padding_info = torch.randint(0, 10, (3, 2))
            data_samples = [DataSample() for _ in range(2)]
            update_fn(padding_info, None, data_samples)

        padding_info = torch.randint(0, 10, (3, 2))
        channel_order = dict(aaa='RGB', BBB='single')
        output = update_fn(padding_info, channel_order)
        self.assertEqual(len(output), 3)
        for data, padding_tar in zip(output, padding_info):
            self.assertTrue(
                (data.metainfo['padding_size'] == padding_tar).all())
            self.assertEqual(data.metainfo['aaa_output_channel_order'], 'RGB')
            self.assertEqual(data.metainfo['BBB_output_channel_order'],
                             'single')

        # channel_order is None + data_sample is not None (use last output)
        padding_info = torch.randint(0, 10, (3, 2))
        output = update_fn(padding_info, None, output)
        self.assertEqual(len(output), 3)
        for data, padding_tar in zip(output, padding_info):
            self.assertTrue(
                (data.metainfo['padding_size'] == padding_tar).all())
            # channel order in last output remains
            self.assertEqual(data.metainfo['aaa_output_channel_order'], 'RGB')
            self.assertEqual(data.metainfo['BBB_output_channel_order'],
                             'single')

    def test_preprocess_image_tensor(self):
        """We test five cases:

        1. raise error when input is not 4D tensor
        2. no metainfo + output channel order is None: no conversion
        3. no metainfo + output channel order is RGB: do conversion to RGB
        4. metainfo (BGR) + output channel order is RGB: do conversion to RGB
        5. metainfo (single) + output channel order is RGB: no conversion
        6. datasample is None: return new data sample with metainfo
        """
        data_preprocessor = DataPreprocessor()
        process_fn = data_preprocessor._preprocess_image_tensor
        # 1. raise error
        with self.assertRaises(AssertionError):
            process_fn(torch.rand(2, 25), None)
        # [0, 0, 0] for all time
        target_padding_info = torch.FloatTensor([0, 0, 0])

        # 2. no metainfo + output channel order is None
        inputs = torch.randint(0, 255, (4, 3, 5, 5))
        targets = (inputs - 127.5) / 127.5
        data_samples = [
            DataSample(metainfo=dict(mm_img_channel_order='RGB'))
            for _ in range(4)
        ]
        outputs, data_samples = process_fn(inputs, data_samples)
        assert_allclose(outputs, targets)
        for data in data_samples:
            self.assertEqual(data.metainfo['img_output_channel_order'], 'BGR')
            padding_size = data.metainfo['padding_size']
            self.assertEqual(padding_size.shape, (3, ))
            self.assertTrue((padding_size == target_padding_info).all())

        # 3. no metainfo + output channel order is RGB -> do conversion
        data_preprocessor = DataPreprocessor(output_channel_order='RGB')
        process_fn = data_preprocessor._preprocess_image_tensor
        inputs = torch.randint(0, 255, (4, 3, 5, 5))
        targets = ((inputs - 127.5) / 127.5)[:, [2, 1, 0]]
        data_samples = [DataSample() for _ in range(4)]
        outputs, data_samples = process_fn(inputs, data_samples)
        assert_allclose(outputs, targets)
        for data in data_samples:
            self.assertEqual(data.metainfo['img_output_channel_order'], 'RGB')
            padding_size = data.metainfo['padding_size']
            self.assertEqual(padding_size.shape, (3, ))
            self.assertTrue((padding_size == target_padding_info).all())

        # 4. metainfo (BGR) + output channel order is RGB -> do conversion
        data_preprocessor = DataPreprocessor(output_channel_order='RGB')
        process_fn = data_preprocessor._preprocess_image_tensor
        inputs = torch.randint(0, 255, (4, 3, 5, 5))
        targets = ((inputs - 127.5) / 127.5)[:, [2, 1, 0]]
        data_samples = [
            DataSample(metainfo=dict(gt_channel_order='BGR')) for _ in range(4)
        ]
        outputs, data_samples = process_fn(inputs, data_samples, 'gt_img')
        assert_allclose(outputs, targets)
        for data in data_samples:
            self.assertEqual(data.metainfo['gt_img_output_channel_order'],
                             'RGB')
            padding_size = data.metainfo['padding_size']
            self.assertEqual(padding_size.shape, (3, ))
            self.assertTrue((padding_size == target_padding_info).all())

        # 5. metainfo (single) + output channel order is RGB -> no conversion
        data_preprocessor = DataPreprocessor(output_channel_order='RGB')
        process_fn = data_preprocessor._preprocess_image_tensor
        inputs = torch.randint(0, 255, (4, 1, 5, 5))
        targets = ((inputs - 127.5) / 127.5)
        data_samples = [
            DataSample(metainfo=dict(sin_channel_order='single'))
            for _ in range(4)
        ]
        outputs, data_samples = process_fn(inputs, data_samples, 'sin')
        assert_allclose(outputs, targets)
        for data in data_samples:
            self.assertEqual(data.metainfo['sin_output_channel_order'],
                             'single')
            padding_size = data.metainfo['padding_size']
            self.assertEqual(padding_size.shape, (3, ))
            self.assertTrue((padding_size == target_padding_info).all())

        # 6. data sample is None + video inputs
        data_preprocessor = DataPreprocessor(output_channel_order='RGB')
        inputs = torch.randint(0, 255, (4, 5, 3, 5, 5))
        targets = ((inputs - 127.5) / 127.5)[:, :, [2, 1, 0]]
        outputs, data_samples = process_fn(inputs, None, 'sin')
        for data in data_samples:
            self.assertEqual(data.metainfo['sin_output_channel_order'], 'RGB')
            self.assertTrue(
                (data.metainfo['padding_size'] == target_padding_info).all())

    def test_preprocess_image_list(self):
        """We test four cases:

        1. no metainfo + output channel order is None: no conversion
        2. no metainfo + output channel order is RGB + padding:
            do conversion to RGB
        3. metainfo (RGB) + output channel order is RGB + padding:
            no conversion
        4. metainfo (single) + output channel order is RGB: no conversion
        5. data_sample is None
        """
        # no metainfo + output channel order is None
        data_preprocessor = DataPreprocessor()
        process_fn = data_preprocessor._preprocess_image_list
        input1 = torch.randint(0, 255, (3, 3, 5))
        input2 = torch.randint(0, 255, (3, 3, 5))
        inputs = [input1, input2]
        data_samples = [DataSample() for _ in range(2)]
        target = torch.stack([(input1 - 127.5) / 127.5,
                              (input2 - 127.5) / 127.5],
                             dim=0)
        target_padding_info = torch.FloatTensor([0, 0, 0])
        outputs, data_samples = process_fn(inputs, data_samples)
        assert_allclose(outputs, target)
        for data in data_samples:
            self.assertEqual(data.metainfo['img_output_channel_order'], 'BGR')
            self.assertTrue(
                (data.metainfo['padding_size'] == target_padding_info).all())

        # no metainfo + output channel order is RGB
        data_preprocessor = DataPreprocessor(
            output_channel_order='RGB', pad_value=42)
        process_fn = data_preprocessor._preprocess_image_list
        input1 = torch.randint(0, 255, (3, 3, 5))
        input2 = torch.randint(0, 255, (3, 5, 5))
        inputs = [input1, input2]
        data_samples = [DataSample() for _ in range(2)]
        target1 = F.pad(input1.clone(), (0, 0, 0, 2), value=42)  # pad H
        target2 = input2.clone()
        target = torch.stack([target1, target2], dim=0)[:, [2, 1, 0]]
        target = (target - 127.5) / 127.5
        target_padding_info = [
            torch.FloatTensor([0, 2, 0]),
            torch.FloatTensor([0, 0, 0])
        ]
        outputs, data_samples = process_fn(inputs, data_samples)
        assert_allclose(outputs, target)
        for data, pad in zip(data_samples, target_padding_info):
            self.assertEqual(data.metainfo['img_output_channel_order'], 'RGB')
            self.assertTrue((data.metainfo['padding_size'] == pad).all())

        # meta info + output channel order is RGB
        data_preprocessor = DataPreprocessor(
            output_channel_order='RGB', pad_value=42)
        process_fn = data_preprocessor._preprocess_image_list
        input1 = torch.randint(0, 255, (3, 3, 5))
        input2 = torch.randint(0, 255, (3, 5, 3))
        inputs = [input1, input2]
        data_samples = [
            DataSample(metainfo=dict(gt_channel_order='RGB')) for _ in range(2)
        ]
        target1 = F.pad(input1.clone(), (0, 0, 0, 2), value=42)  # pad H
        target2 = F.pad(input2.clone(), (0, 2, 0, 0), value=42)  # pad W
        target = torch.stack([target1, target2], dim=0)
        target = (target - 127.5) / 127.5
        target_padding_info = [
            torch.FloatTensor([0, 2, 0]),
            torch.FloatTensor([0, 0, 2])
        ]
        outputs, data_samples = process_fn(inputs, data_samples, 'gt_img')
        assert_allclose(outputs, target)
        for data, pad in zip(data_samples, target_padding_info):
            self.assertEqual(data.metainfo['gt_img_output_channel_order'],
                             'RGB')
            self.assertTrue((data.metainfo['padding_size'] == pad).all())

        # meta info (single) + output channel order is RGB
        data_preprocessor = DataPreprocessor(
            output_channel_order='RGB', pad_value=42)
        process_fn = data_preprocessor._preprocess_image_list
        input1 = torch.randint(0, 255, (1, 3, 5))
        input2 = torch.randint(0, 255, (1, 5, 3))
        inputs = [input1, input2]
        data_samples = [
            DataSample(metainfo=dict(AA_channel_order='single'))
            for _ in range(2)
        ]
        target1 = F.pad(input1.clone(), (0, 0, 0, 2), value=42)  # pad H
        target2 = F.pad(input2.clone(), (0, 2, 0, 0), value=42)  # pad W
        target = torch.stack([target1, target2], dim=0)
        target = (target - 127.5) / 127.5
        target_padding_info = [
            torch.FloatTensor([0, 2, 0]),
            torch.FloatTensor([0, 0, 2])
        ]
        outputs, data_samples = process_fn(inputs, data_samples, 'AA')
        assert_allclose(outputs, target)
        for data, pad in zip(data_samples, target_padding_info):
            self.assertEqual(data.metainfo['AA_output_channel_order'],
                             'single')
            self.assertTrue((data.metainfo['padding_size'] == pad).all())

        # test data sample is None
        data_preprocessor = DataPreprocessor(
            output_channel_order='RGB', pad_value=42)
        process_fn = data_preprocessor._preprocess_image_list
        input1 = torch.randint(0, 255, (1, 3, 5))
        input2 = torch.randint(0, 255, (1, 5, 3))
        inputs = [input1, input2]
        target1 = F.pad(input1.clone(), (0, 0, 0, 2), value=42)  # pad H
        target2 = F.pad(input2.clone(), (0, 2, 0, 0), value=42)  # pad W
        target = torch.stack([target1, target2], dim=0)
        target = (target - 127.5) / 127.5
        target_padding_info = [
            torch.FloatTensor([0, 2, 0]),
            torch.FloatTensor([0, 0, 2])
        ]
        outputs, data_samples = process_fn(inputs, None, 'test')
        assert_allclose(outputs, target)
        for data, pad in zip(data_samples, target_padding_info):
            self.assertEqual(data.metainfo['test_output_channel_order'],
                             'single')
            self.assertTrue((data.metainfo['padding_size'] == pad).all())

    def test_preprocess_dict_inputs(self):
        """Since preprocess of dict inputs are based on
        `_preprocess_image_list` and `_preprocess_image_tensor`, we just test a
        simple case for translation model and padding behavior."""
        data_preprocessor = DataPreprocessor(output_channel_order='RGB')
        process_fn = data_preprocessor._preprocess_dict_inputs

        inputs = dict(
            img_A=[torch.randint(0, 255, (3, 5, 5)) for _ in range(3)],
            img_B=[torch.randint(0, 255, (3, 5, 5)) for _ in range(3)],
            noise=[torch.randn(16) for _ in range(3)],
            num_batches=3,
            tensor=torch.randint(0, 255, (3, 4, 5, 5)),
            mode=['ema', 'ema', 'ema'],
        )
        data_samples = [
            DataSample(
                metainfo=dict(
                    img_A_channel_order='RGB', img_B_channel_order='BGR'))
            for _ in range(3)
        ]
        target_A = (torch.stack(inputs['img_A']) - 127.5) / 127.5
        target_B = (torch.stack(inputs['img_B']) - 127.5) / 127.5
        target_B = target_B[:, [2, 1, 0]]
        target_noise = torch.stack(inputs['noise'])
        # no metainfo, parse as BGR, do conversion
        target_tensor = ((inputs['tensor'] - 127.5) / 127.5)[:, [2, 1, 0, 3]]

        outputs, data_samples = process_fn(inputs, data_samples)
        assert_allclose(outputs['img_A'], target_A)
        assert_allclose(outputs['img_B'], target_B)
        assert_allclose(outputs['noise'], target_noise)
        assert_allclose(outputs['tensor'], target_tensor)
        self.assertEqual(outputs['mode'], 'ema')
        self.assertEqual(outputs['num_batches'], 3)

        # test no padding checking --> no tensor inputs
        inputs = dict(num_batches=3, mode=['ema', 'ema', 'ema'])
        outputs, data_samples = process_fn(inputs, data_samples)
        self.assertEqual(outputs['mode'], 'ema')
        self.assertEqual(outputs['num_batches'], 3)

        # test error in padding checking
        data_samples = [DataSample() for _ in range(2)]
        inputs = dict(
            img_A=[
                torch.randint(0, 255, (3, 3, 5)),
                torch.randint(0, 255, (3, 5, 5))
            ],
            img_B=[
                torch.randint(0, 255, (3, 5, 5)),
                torch.randint(0, 255, (3, 5, 5))
            ],
        )
        with self.assertRaises(ValueError):
            process_fn(inputs, data_samples)

    def test_prerprocess_data_sample(self):
        """Only test training and testint mode in this test case."""
        data_preprocessor = DataPreprocessor(
            data_keys=['gt_img', 'AA', 'dk'], output_channel_order='RGB')
        cov_fn = data_preprocessor._preprocess_data_sample

        # test training is True
        metainfo = dict(gt_channel_order='RGB', tar_channel_order='BGR')
        data_samples = [
            DataSample(
                gt_img=torch.randint(0, 255, (3, 5, 5)),
                AA=torch.randint(0, 255, (3, 5, 5)),
                metainfo=metainfo,
            ) for _ in range(3)
        ]
        tar_AA = torch.stack([data.AA for data in data_samples])
        tar_AA = ((tar_AA - 127.5) / 127.5)[:, [2, 1, 0]]
        tar_gt = torch.stack([data.gt_img for data in data_samples])
        tar_gt = (tar_gt - 127.5) / 127.5
        outputs = cov_fn(data_samples, True)
        assert_allclose(outputs.gt_img, tar_gt)
        assert_allclose(outputs.AA, tar_AA)

        # test training is False
        data_samples = [
            DataSample(
                gt_img=torch.randint(0, 255, (3, 5, 5)),
                AA=torch.randint(0, 255, (3, 5, 5)),
                metainfo=metainfo,
            ) for _ in range(3)
        ]
        tar_AA = torch.stack([data.AA for data in data_samples])
        tar_AA = tar_AA  # BGR, no conversion
        tar_gt = torch.stack([data.gt_img for data in data_samples])
        tar_gt = tar_gt[:, [2, 1, 0]]  # RGB -> BGR
        outputs = cov_fn(data_samples, False)
        assert_allclose(outputs.gt_img, tar_gt)
        assert_allclose(outputs.AA, tar_AA)

        # test no stack
        data_preprocessor.stack_data_sample = False
        data_samples = [
            DataSample(
                gt_img=torch.randint(0, 255, (3, 5, 5)),
                AA=torch.randint(0, 255, (3, 5, 5)),
                metainfo=metainfo,
            ) for _ in range(3)
        ]
        tar_AA = torch.stack([data.AA for data in data_samples])
        tar_AA = tar_AA  # BGR, no conversion
        tar_gt = torch.stack([data.gt_img for data in data_samples])
        tar_gt = tar_gt[:, [2, 1, 0]]  # RGB -> BGR
        outputs = cov_fn(data_samples, False)
        output_gt = torch.stack([out.gt_img for out in outputs])
        output_AA = torch.stack([out.AA for out in outputs])
        assert_allclose(output_gt, tar_gt)
        assert_allclose(output_AA, tar_AA)

    def test_destruct_tensor_norm_and_conversion(self):
        """Test batch inputs, single input and batch inputs + no norm in this
        unit test."""
        data_preprocessor = DataPreprocessor()
        cov_fn = data_preprocessor._destruct_norm_and_conversion

        # test batch
        metainfo = dict(
            img_output_channel_order='RGB',
            aaa_output_channel_order='single',
            bbb_output_channel_order='BGR',
        )
        img = torch.randn([3, 3, 5, 5])
        aaa = torch.randn([3, 1, 5, 5])
        bbb = torch.randn([3, 3, 5, 5])
        ccc = torch.randn([3, 3, 5, 5])
        data_samples = [DataSample(metainfo=metainfo) for _ in range(3)]
        tar_img = ((img * 127.5) + 127.5)[:, [2, 1, 0]]
        tar_aaa = (aaa * 127.5) + 127.5
        tar_bbb = (bbb * 127.5) + 127.5
        tar_ccc = (ccc * 127.5) + 127.5
        assert_allclose(cov_fn(img, data_samples, 'img'), tar_img)
        assert_allclose(cov_fn(aaa, data_samples, 'aaa'), tar_aaa)
        assert_allclose(cov_fn(bbb, data_samples, 'bbb'), tar_bbb)
        assert_allclose(cov_fn(ccc, data_samples, 'bbb'), tar_ccc)

        # test single sample
        img = torch.randn([3, 5, 5])
        aaa = torch.randn([1, 5, 5])
        bbb = torch.randn([3, 5, 5])
        ccc = torch.randn([3, 5, 5])
        data_samples = DataSample(metainfo=metainfo)
        tar_img = ((img * 127.5) + 127.5)[[2, 1, 0], ...]
        tar_aaa = (aaa * 127.5) + 127.5
        tar_bbb = (bbb * 127.5) + 127.5
        tar_ccc = (ccc * 127.5) + 127.5
        assert_allclose(cov_fn(img, data_samples, 'img'), tar_img)
        assert_allclose(cov_fn(aaa, data_samples, 'aaa'), tar_aaa)
        assert_allclose(cov_fn(bbb, data_samples, 'bbb'), tar_bbb)
        assert_allclose(cov_fn(ccc, data_samples, 'bbb'), tar_ccc)

        # test no norm
        data_preprocessor._enable_normalize = False
        metainfo = dict(img_output_channel_order='RGB', )
        img = torch.randn([3, 3, 5, 5])
        data_samples = [DataSample(metainfo=metainfo) for _ in range(3)]
        tar_img = img[:, [2, 1, 0]]
        assert_allclose(cov_fn(img, data_samples, 'img'), tar_img)

    def test_destruct_tensor_padding(self):
        """Test five cases in this unit test.

        1. batch inputs + same padidng = True
        2. batch inputs + same padding = False
        3. single inputs
        4. data_samples is None
        5. padding_size is not found in metainfo
        6. one data sample + padding size is not found in metainfo
        7. list of sample + padding size is not found in metainfo

        8. data sample is stacked
        9. data sample is stacked + padding size not found in metainfo
        """
        data_preprocessor = DataPreprocessor()
        cov_fn = data_preprocessor._destruct_padding

        # data sample is None, no un-padding
        batch_tensor = torch.randint(0, 255, (2, 3, 3, 3))
        tar_output = batch_tensor.clone()
        output = cov_fn(batch_tensor, None)
        assert_allclose(tar_output, output)

        # batch inputs + same_padding is True
        batch_tensor = torch.randint(0, 255, (2, 3, 5, 5))
        metainfo_1 = dict(padding_size=torch.FloatTensor([2, 0]))
        metainfo_2 = dict(padding_size=torch.FloatTensor([0, 2]))
        data_samples = [
            DataSample(metainfo=metainfo_1),
            DataSample(metainfo=metainfo_2)
        ]
        output = cov_fn(batch_tensor, data_samples)
        self.assertEqual(output.shape, (2, 3, 3, 5))

        # batch inputs + same padding is False
        batch_tensor = torch.randint(0, 255, (2, 3, 5, 5))
        metainfo_1 = dict(padding_size=torch.FloatTensor([2, 0]))
        metainfo_2 = dict(padding_size=torch.FloatTensor([0, 2]))
        data_samples = [
            DataSample(metainfo=metainfo_1),
            DataSample(metainfo=metainfo_2)
        ]
        output = cov_fn(batch_tensor, data_samples, False)
        self.assertIsInstance(output, list)
        self.assertEqual(output[0].shape, (3, 3, 5))
        self.assertEqual(output[1].shape, (3, 5, 3))

        # single inputs
        batch_tensor = torch.randint(0, 255, (3, 5, 5))
        data_samples = DataSample(metainfo=metainfo_1)
        output = cov_fn(batch_tensor, data_samples, False)
        self.assertEqual(output.shape, (3, 3, 5))

        # single data sample + test metainfo not found
        batch_tensor = torch.randint(0, 255, (3, 5, 5))
        data_preprocessor._done_padding = False
        output = cov_fn(batch_tensor, DataSample(), False)
        assert_allclose(output, batch_tensor)

        # list of data sample + test metainfo not found
        batch_tensor = torch.randint(0, 255, (2, 3, 5, 5))
        output = cov_fn(batch_tensor, [DataSample()] * 2, False)
        assert_allclose(output, batch_tensor)

        data_preprocessor._done_padding = True
        output = cov_fn(batch_tensor, DataSample(), False)
        assert_allclose(output, batch_tensor)

        # test stacked data sample
        data_samples = [
            DataSample(metainfo=metainfo_1),
            DataSample(metainfo=metainfo_2)
        ]
        stacked_data_sample = DataSample.stack(data_samples)
        batch_tensor = torch.randint(0, 255, (2, 3, 5, 5))
        output = cov_fn(batch_tensor, stacked_data_sample, False)
        self.assertIsInstance(output, list)
        self.assertEqual(output[0].shape, (3, 3, 5))
        self.assertEqual(output[1].shape, (3, 5, 3))

        # test stacked data sample + metainfo is None
        stacked_data_sample = DataSample()
        batch_tensor = torch.randint(0, 255, (2, 3, 5, 5))
        output = cov_fn(batch_tensor, stacked_data_sample, True)
        self.assertEqual(output.shape, (2, 3, 5, 5))

    def test_forward(self):
        """Test five cases in this unit test.

        We do not cover all possible
        cases, as we test them in test cases for other functions.
        1. Input is tensor.
        2. Input is list of tensors.
        3. Input is a list of dict.
        4. Input is dict.
        5. Input is tensor + no norm
        6. data samples behavior in training = False / True
        7. Input is wrong type.
        """
        data_preprocessor = DataPreprocessor()

        # 1. input is tensor
        inputs = torch.randint(0, 255, (2, 3, 5, 5))
        data = dict(inputs=inputs)
        tar_output = (inputs.clone() - 127.5) / 127.5
        data_preprocessor = DataPreprocessor()
        data = data_preprocessor(data)
        assert_allclose(data['inputs'], tar_output)

        # 2. input is list of tensor
        input1 = torch.randn(3, 3, 5)
        input2 = torch.randn(3, 3, 5)

        data = dict(inputs=[input1, input2])

        data = data_preprocessor(data)

        self.assertEqual(data['inputs'].shape, (2, 3, 3, 5))

        target_input1 = (input1.clone() - 127.5) / 127.5
        target_input2 = (input2.clone() - 127.5) / 127.5
        assert_allclose(target_input1, data['inputs'][0])
        assert_allclose(target_input2, data['inputs'][1])

        # 3. input is list of dict
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
        data_preprocessor = DataPreprocessor(output_channel_order='RGB')
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

        # 4. nput is dict
        sampler_results = dict(inputs=dict(num_batches=2, mode='ema'))
        data = data_preprocessor(sampler_results)
        self.assertEqual(data['inputs'], sampler_results['inputs'])
        self.assertIsNone(data['data_samples'])

        # test dict input with tensor
        sampler_results = dict(inputs=dict(fake_img=torch.randn(2, 3, 10, 10)))
        data = data_preprocessor(sampler_results)

        # # 5. input is tensor + no norm
        data_preprocessor = DataPreprocessor(mean=None, std=None)
        input1 = torch.randn(3, 3, 5)
        input2 = torch.randn(3, 3, 5)
        data = dict(inputs=torch.stack([input1, input2], dim=0))
        data = data_preprocessor(data)
        self.assertEqual(data['inputs'].shape, (2, 3, 3, 5))
        self.assertTrue((data['inputs'] == torch.stack([input1, input2],
                                                       dim=0)).all())

        # 6. data samples behavior in training = False / True
        data_preprocessor = DataPreprocessor()
        gt_inp1 = torch.randint(0, 255, (3, 5, 5))
        gt_inp2 = torch.randint(0, 255, (3, 5, 5))

        data_samples = [DataSample(gt_img=gt_inp1), DataSample(gt_img=gt_inp2)]
        data = dict(inputs=[input1, input2], data_samples=data_samples)
        data = data_preprocessor(data, training=False)
        assert_allclose(data['data_samples'].gt_img,
                        torch.stack([gt_inp1, gt_inp2]))

        data = dict(inputs=[input1, input2], data_samples=data_samples)
        data = data_preprocessor(data, training=True)
        self.assertTrue((data['data_samples'].gt_img <= 1).all())

        # 7. Input is wrong type
        data = dict(inputs='wrong type')
        with self.assertRaises(ValueError):
            data = data_preprocessor(data)

    def test_destruct(self):
        """We test the following cases in this unit test:

        1. test un-padding
        2. test un-norm + no un-padding
        3. test no un-norm + no un-padding
        """
        # 1. test un-padding
        data_preprocessor = DataPreprocessor()
        input1 = torch.randn(3, 3, 5)
        input2 = torch.randn(3, 5, 3)
        data = dict(inputs=[input1, input2])
        data = data_preprocessor(data)
        self.assertEqual(data['inputs'].shape, (2, 3, 5, 5))
        tar_pad_size = torch.FloatTensor([[0, 2, 0], [0, 0, 2]])
        padding_sizes = data['data_samples'].metainfo['padding_size']
        for padding_size, tar in zip(padding_sizes, tar_pad_size):
            assert_allclose(padding_size, tar)
        destruct_batch = data_preprocessor.destruct(data['inputs'],
                                                    data['data_samples'])
        self.assertEqual(destruct_batch.shape, (2, 3, 3, 5))

        # 2. test no un-padding + un-norm
        data_preprocessor = DataPreprocessor()
        input1 = torch.randint(0, 255, (3, 5, 5))
        input2 = torch.randint(0, 255, (3, 5, 5))
        data = dict(inputs=[input1, input2])
        data = data_preprocessor(data)
        self.assertEqual(data['inputs'].shape, (2, 3, 5, 5))
        tar_pad_size = torch.FloatTensor([[0, 0, 0], [0, 0, 0]])
        padding_sizes = data['data_samples'].metainfo['padding_size']
        for padding_size, tar in zip(padding_sizes, tar_pad_size):
            assert_allclose(padding_size, tar)
        destruct_batch = data_preprocessor.destruct(data['inputs'],
                                                    data['data_samples'])
        self.assertEqual(destruct_batch.shape, (2, 3, 5, 5))
        assert_allclose(destruct_batch,
                        torch.stack([input1, input2], dim=0).float())

        # 3. test no un-norm
        data_preprocessor = DataPreprocessor(std=None, mean=None)
        input1 = torch.randint(0, 255, (1, 5, 5))
        input2 = torch.randint(0, 255, (1, 5, 5))
        inputs = torch.stack([input1, input2], dim=0)
        destruct_batch = data_preprocessor.destruct(inputs)
        self.assertEqual(destruct_batch.shape, (2, 1, 5, 5))
        assert_allclose(destruct_batch.float(), inputs.float())


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
