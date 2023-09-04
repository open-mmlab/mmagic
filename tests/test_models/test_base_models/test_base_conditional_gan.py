# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from unittest import TestCase
from unittest.mock import MagicMock

import torch
from mmengine import MessageHub
from mmengine.structures import LabelData
from mmengine.testing import assert_allclose
from torch.nn import ModuleList

from mmagic.models import BaseConditionalGAN, DataPreprocessor
from mmagic.models.losses import (DiscShiftLossComps, GANLossComps,
                                  GeneratorPathRegularizerComps,
                                  GradientPenaltyLossComps)
from mmagic.structures import DataSample

generator = dict(
    type='SAGANGenerator',
    output_scale=32,
    base_channels=32,
    attention_cfg=dict(type='SelfAttentionBlock'),
    attention_after_nth_block=2,
    with_spectral_norm=True)
discriminator = dict(
    type='ProjDiscriminator',
    input_scale=32,
    base_channels=32,
    attention_cfg=dict(type='SelfAttentionBlock'),
    attention_after_nth_block=1,
    with_spectral_norm=True)


class ToyCGAN(BaseConditionalGAN):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_generator(self, inputs, data_samples, optimizer_wrapper):
        return dict(loss_gen=1)

    def train_discriminator(self, inputs, data_samples, optimizer_wrapper):
        return dict(loss_disc=2)


class TestBaseGAN(TestCase):

    def test_val_step_and_test_step(self):
        gan = ToyCGAN(
            noise_size=10,
            num_classes=10,
            generator=deepcopy(generator),
            data_preprocessor=DataPreprocessor())
        gan.eval()

        # no mode
        inputs = dict(inputs=dict(num_batches=3))
        outputs_val = gan.val_step(inputs)
        outputs_test = gan.test_step(inputs)
        self.assertEqual(len(outputs_val), 3)
        self.assertEqual(len(outputs_test), 3)
        for out_val, out_test in zip(outputs_val, outputs_test):
            self.assertEqual(out_val.fake_img.shape, (3, 32, 32))
            self.assertEqual(out_test.fake_img.shape, (3, 32, 32))

        # set mode
        inputs = dict(inputs=dict(num_batches=4, sample_model='orig'))
        outputs_val = gan.val_step(inputs)
        outputs_test = gan.test_step(inputs)
        self.assertEqual(len(outputs_val), 4)
        self.assertEqual(len(outputs_test), 4)
        for out_val, out_test in zip(outputs_val, outputs_test):
            self.assertEqual(out_val.sample_model, 'orig')
            self.assertEqual(out_test.sample_model, 'orig')
            self.assertEqual(out_val.fake_img.shape, (3, 32, 32))
            self.assertEqual(out_test.fake_img.shape, (3, 32, 32))

        inputs = dict(inputs=dict(num_batches=4, sample_model='orig/ema'))
        self.assertRaises(AssertionError, gan.val_step, inputs)

        inputs = dict(inputs=dict(num_batches=4, sample_model='ema'))
        self.assertRaises(AssertionError, gan.val_step, inputs)

        # set noise and label input
        gt_label = torch.randint(0, 10, (1, ))
        inputs = dict(
            inputs=dict(noise=torch.randn(1, 10)),
            data_samples=[DataSample(gt_label=LabelData(label=gt_label))])
        outputs_val = gan.val_step(inputs)
        outputs_test = gan.test_step(inputs)
        self.assertEqual(len(outputs_val), 1)
        self.assertEqual(len(outputs_val), 1)
        for idx in range(1):
            test_fake_img = outputs_test[idx].fake_img
            val_fake_img = outputs_val[idx].fake_img
            test_label = outputs_test[idx].gt_label.label
            val_label = outputs_val[idx].gt_label.label
            self.assertEqual(test_label, gt_label)
            self.assertEqual(val_label, gt_label)
            assert_allclose(test_fake_img, val_fake_img)

    def test_forward(self):
        # set a gan w/o EMA
        gan = ToyCGAN(
            noise_size=10,
            num_classes=10,
            generator=deepcopy(generator),
            data_preprocessor=DataPreprocessor())
        gan.eval()
        inputs = dict(num_batches=3)
        outputs = gan(inputs, None)
        self.assertEqual(len(outputs), 3)
        for out in outputs:
            self.assertEqual(out.fake_img.shape, (3, 32, 32))

        outputs = gan(inputs)
        self.assertEqual(len(outputs), 3)
        for out in outputs:
            self.assertEqual(out.fake_img.shape, (3, 32, 32))

        outputs = gan(torch.randn(3, 10))
        self.assertEqual(len(outputs), 3)
        for out in outputs:
            self.assertEqual(out.fake_img.shape, (3, 32, 32))

        # set a gan w EMA
        gan = ToyCGAN(
            noise_size=10,
            num_classes=10,
            generator=deepcopy(generator),
            data_preprocessor=DataPreprocessor(),
            ema_config=dict(interval=1))
        gan.eval()
        # inputs = dict(inputs=dict(num_batches=3))
        inputs = dict(num_batches=3)
        outputs = gan(inputs)
        self.assertEqual(len(outputs), 3)
        for out in outputs:
            self.assertEqual(out.fake_img.shape, (3, 32, 32))

        # inputs = dict(inputs=dict(num_batches=3, sample_model='ema/orig'))
        inputs = dict(num_batches=3, sample_model='ema/orig')
        outputs = gan(inputs)
        self.assertEqual(len(outputs), 3)
        for out in outputs:
            ema_img = out.ema
            orig_img = out.orig
            self.assertEqual(ema_img.fake_img.shape, orig_img.fake_img.shape)
            self.assertTrue(out.sample_model, 'ema/orig')

        # inputs = dict(inputs=dict(noise=torch.randn(4, 10)))
        inputs = dict(noise=torch.randn(4, 10))
        outputs = gan(inputs)
        self.assertEqual(len(outputs), 4)
        for out in outputs:
            self.assertEqual(out.fake_img.shape, (3, 32, 32))

        # test data sample input
        # inputs = dict(inputs=dict(noise=torch.randn(3, 10)))
        inputs = dict(noise=torch.randn(3, 10))
        label = [torch.randint(0, 10, (1, )) for _ in range(3)]
        data_sample = [DataSample() for _ in range(3)]
        for idx, sample in enumerate(data_sample):
            sample.set_gt_label(label[idx])
        print(DataSample.stack(data_sample))
        outputs = gan(inputs, DataSample.stack(data_sample))
        self.assertEqual(len(outputs), 3)
        for idx, output in enumerate(outputs):
            self.assertEqual(output.gt_label.label, label[idx])

    def test_custom_loss(self):
        message_hub = MessageHub.get_instance('basegan-test-custom-loss')
        message_hub.update_info('iter', 10)

        gan_loss = dict(type='GANLossComps', gan_type='vanilla')

        # test loss config is dict()
        gan = BaseConditionalGAN(
            noise_size=10,
            num_classes=10,
            generator=deepcopy(generator),
            discriminator=deepcopy(discriminator),
            data_preprocessor=DataPreprocessor(),
            loss_config=dict())
        self.assertIsNone(gan.gan_loss)
        self.assertIsNone(gan.disc_auxiliary_losses)
        self.assertIsNone(gan.gen_auxiliary_losses)

        # test loss config is list
        disc_auxiliary_loss_list = [
            dict(type='DiscShiftLossComps'),
            dict(type='GradientPenaltyLossComps')
        ]
        gen_auxiliary_loss_list = [dict(type='GeneratorPathRegularizerComps')]
        loss_config = dict(
            gan_loss=gan_loss,
            disc_auxiliary_loss=disc_auxiliary_loss_list,
            gen_auxiliary_loss=gen_auxiliary_loss_list)
        gan = BaseConditionalGAN(
            noise_size=10,
            num_classes=10,
            generator=deepcopy(generator),
            discriminator=deepcopy(discriminator),
            data_preprocessor=DataPreprocessor(),
            loss_config=loss_config)
        self.assertIsInstance(gan.disc_auxiliary_losses, ModuleList)
        self.assertIsInstance(gan.disc_auxiliary_losses[0], DiscShiftLossComps)
        self.assertIsInstance(gan.disc_auxiliary_losses[1],
                              GradientPenaltyLossComps)
        self.assertIsInstance(gan.gen_auxiliary_losses, ModuleList)
        self.assertIsInstance(gan.gen_auxiliary_losses[0],
                              GeneratorPathRegularizerComps)

        # test loss config is single dict
        disc_auxiliary_loss = dict(
            type='DiscShiftLossComps', data_info=dict(pred='disc_pred_fake'))
        gen_auxiliary_loss = dict(
            type='GeneratorPathRegularizerComps',
            data_info=dict(generator='gen', num_batches='batch_size'))
        loss_config = dict(
            gan_loss=gan_loss,
            disc_auxiliary_loss=disc_auxiliary_loss,
            gen_auxiliary_loss=gen_auxiliary_loss)

        gan = BaseConditionalGAN(
            noise_size=10,
            num_classes=10,
            generator=deepcopy(generator),
            discriminator=deepcopy(discriminator),
            data_preprocessor=DataPreprocessor(),
            loss_config=loss_config)
        self.assertIsInstance(gan.gan_loss, GANLossComps)
        self.assertIsInstance(gan.disc_auxiliary_losses, ModuleList)
        self.assertIsInstance(gan.disc_auxiliary_losses[0], DiscShiftLossComps)
        self.assertIsInstance(gan.gen_auxiliary_losses, ModuleList)
        self.assertIsInstance(gan.gen_auxiliary_losses[0],
                              GeneratorPathRegularizerComps)

        # test forward custom loss terms
        gan = BaseConditionalGAN(
            noise_size=10,
            num_classes=10,
            generator=deepcopy(generator),
            discriminator=deepcopy(discriminator),
            data_preprocessor=DataPreprocessor(),
            loss_config=loss_config)
        # mock gen aux loss to avoid build styleGAN Generator
        gen_aux_loss_mock = MagicMock(return_value=torch.Tensor([1.]))
        gen_aux_loss_mock.loss_name = MagicMock(return_value='loss_gen_aux')
        gan._modules['gen_auxiliary_losses'] = [gen_aux_loss_mock]
        # mock optim wrapper
        optimizer_wrapper = {
            'discriminator': MagicMock(),
            'generator': MagicMock()
        }
        optimizer_wrapper['discriminator']._accumulative_counts = 1
        optimizer_wrapper['generator']._accumulative_counts = 1

        inputs = dict(img=torch.randn(3, 3, 32, 32))
        label = [torch.randint(0, 10, (1, )) for _ in range(3)]
        data_sample = [DataSample() for _ in range(3)]
        for idx, sample in enumerate(data_sample):
            sample.set_gt_label(label[idx])
        data = dict(inputs=inputs, data_samples=data_sample)

        log_vars = gan.train_step(data, optim_wrapper=optimizer_wrapper)
        self.assertIn('loss', log_vars)
        self.assertIn('loss_disc_fake', log_vars)
        self.assertIn('loss_disc_real', log_vars)
        self.assertIn('loss_disc_fake_g', log_vars)
        self.assertIn('loss_gen_aux', log_vars)
        self.assertIn('loss_disc_shift', log_vars)

        # test forward with only gan loss
        loss_config = dict(gan_loss=gan_loss)
        gan = BaseConditionalGAN(
            noise_size=10,
            num_classes=10,
            generator=deepcopy(generator),
            discriminator=deepcopy(discriminator),
            data_preprocessor=DataPreprocessor(),
            loss_config=loss_config)
        log_vars = gan.train_step(data, optim_wrapper=optimizer_wrapper)
        self.assertIn('loss', log_vars)
        self.assertIn('loss_disc_fake', log_vars)
        self.assertIn('loss_disc_real', log_vars)
        self.assertIn('loss_disc_fake_g', log_vars)
        self.assertNotIn('loss_gen_aux', log_vars)
        self.assertNotIn('loss_disc_shift', log_vars)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
