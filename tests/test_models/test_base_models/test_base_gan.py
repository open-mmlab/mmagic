# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from unittest import TestCase
from unittest.mock import MagicMock

import torch
from mmengine import MessageHub
from mmengine.optim import OptimWrapper, OptimWrapperDict
from mmengine.testing import assert_allclose
from torch.nn import ModuleList
from torch.optim import SGD

from mmagic.models import BaseGAN, DataPreprocessor
from mmagic.models.losses import (DiscShiftLossComps, GANLossComps,
                                  GeneratorPathRegularizerComps,
                                  GradientPenaltyLossComps)
from mmagic.registry import MODELS
from mmagic.structures import DataSample

generator = dict(type='DCGANGenerator', output_scale=8, base_channels=8)
discriminator = dict(
    type='DCGANDiscriminator',
    base_channels=8,
    input_scale=8,
    output_scale=4,
    out_channels=1)


class ToyGAN(BaseGAN):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_generator(self, inputs, data_samples, optimizer_wrapper):
        return dict(loss_gen=1)

    def train_discriminator(self, inputs, data_samples, optimizer_wrapper):
        return dict(loss_disc=2)


class TestBaseGAN(TestCase):

    def test_init(self):
        gan = ToyGAN(
            noise_size=5,
            generator=deepcopy(generator),
            discriminator=deepcopy(discriminator),
            data_preprocessor=DataPreprocessor())
        self.assertIsInstance(gan, BaseGAN)
        self.assertIsInstance(gan.data_preprocessor, DataPreprocessor)

        # test only generator have noise size
        gen_cfg = deepcopy(generator)
        gen_cfg['noise_size'] = 10
        gan = ToyGAN(
            generator=gen_cfg,
            discriminator=discriminator,
            data_preprocessor=DataPreprocessor())
        self.assertEqual(gan.noise_size, 10)

        # test init with nn.Module
        gen_cfg = deepcopy(generator)
        gen_cfg['noise_size'] = 10
        disc_cfg = deepcopy(discriminator)
        gen = MODELS.build(gen_cfg)
        disc = MODELS.build(disc_cfg)
        gan = ToyGAN(
            generator=gen,
            discriminator=disc,
            data_preprocessor=DataPreprocessor())
        self.assertEqual(gan.generator, gen)
        self.assertEqual(gan.discriminator, disc)

        # test init without discriminator
        gan = ToyGAN(generator=gen, data_preprocessor=DataPreprocessor())
        self.assertEqual(gan.discriminator, None)

        self.assertIsNone(gan.gan_loss)
        self.assertIsNone(gan.gen_auxiliary_losses)
        self.assertIsNone(gan.disc_auxiliary_losses)
        self.assertEqual(gan.loss_config, dict())

    def test_train_step(self):
        # prepare model
        accu_iter = 2
        n_disc = 2
        message_hub = MessageHub.get_instance('basegan-test')
        gan = ToyGAN(
            noise_size=10,
            generator=generator,
            discriminator=discriminator,
            data_preprocessor=DataPreprocessor(),
            discriminator_steps=n_disc)
        ToyGAN.train_discriminator = MagicMock(
            return_value=dict(loss_disc=torch.Tensor(1), loss=torch.Tensor(1)))
        ToyGAN.train_generator = MagicMock(
            return_value=dict(loss_gen=torch.Tensor(2), loss=torch.Tensor(2)))
        # prepare messageHub
        message_hub.update_info('iter', 0)
        # prepare optimizer
        gen_optim = SGD(gan.generator.parameters(), lr=0.1)
        disc_optim = SGD(gan.discriminator.parameters(), lr=0.1)
        optim_wrapper_dict = OptimWrapperDict(
            generator=OptimWrapper(gen_optim, accumulative_counts=accu_iter),
            discriminator=OptimWrapper(
                disc_optim, accumulative_counts=accu_iter))
        # prepare inputs
        inputs = torch.randn(1, 3, 4, 4)
        data = dict(inputs=inputs)

        # simulate train_loop here
        disc_update_times = 0
        for idx in range(n_disc * accu_iter):
            message_hub.update_info('iter', idx)
            log = gan.train_step(data, optim_wrapper_dict)
            if (idx + 1) == n_disc * accu_iter:
                # should update at after (n_disc * accu_iter)
                self.assertEqual(ToyGAN.train_generator.call_count, accu_iter)
                self.assertEqual(
                    set(log.keys()), set(['loss', 'loss_disc', 'loss_gen']))
            else:
                # should not update when discriminator's updating is unfinished
                self.assertEqual(ToyGAN.train_generator.call_count, 0)
                self.assertEqual(log.keys(), set(['loss', 'loss_disc']))

            # disc should update once for each iteration
            disc_update_times += 1
            self.assertEqual(ToyGAN.train_discriminator.call_count,
                             disc_update_times)

    def test_update_ema(self):
        # prepare model
        n_gen = 4
        n_disc = 2
        accu_iter = 2
        ema_interval = 3
        message_hub = MessageHub.get_instance('basegan-test-ema')
        gan = ToyGAN(
            noise_size=10,
            generator=generator,
            discriminator=discriminator,
            data_preprocessor=DataPreprocessor(),
            discriminator_steps=n_disc,
            generator_steps=n_gen,
            ema_config=dict(interval=ema_interval))
        gan.train_discriminator = MagicMock(
            return_value=dict(loss_disc=torch.Tensor(1), loss=torch.Tensor(1)))
        gan.train_generator = MagicMock(
            return_value=dict(loss_gen=torch.Tensor(2), loss=torch.Tensor(2)))

        self.assertTrue(gan.with_ema_gen)
        # mock generator_ema with MagicMock
        del gan.generator_ema
        setattr(gan, 'generator_ema', MagicMock())
        # prepare messageHub
        message_hub.update_info('iter', 0)
        # prepare optimizer
        gen_optim = SGD(gan.generator.parameters(), lr=0.1)
        disc_optim = SGD(gan.discriminator.parameters(), lr=0.1)
        optim_wrapper_dict = OptimWrapperDict(
            generator=OptimWrapper(gen_optim, accumulative_counts=accu_iter),
            discriminator=OptimWrapper(
                disc_optim, accumulative_counts=accu_iter))
        # prepare inputs
        inputs = torch.randn(1, 3, 4, 4)
        data = dict(inputs=inputs)

        # simulate train_loop here
        ema_times = 0
        gen_update_times = 0
        disc_update_times = 0
        for idx in range(n_disc * accu_iter * ema_interval):
            message_hub.update_info('iter', idx)
            gan.train_step(data, optim_wrapper_dict)
            if (idx + 1) % (n_disc * accu_iter) == 0:
                ema_times += 1
                gen_update_times += accu_iter * n_gen

            disc_update_times += 1
            self.assertEqual(gan.generator_ema.update_parameters.call_count,
                             ema_times)
            self.assertEqual(gan.train_generator.call_count, gen_update_times)
            # disc should update once for each iteration
            self.assertEqual(gan.train_discriminator.call_count,
                             disc_update_times)

    def test_val_step_and_test_step(self):
        gan = ToyGAN(
            noise_size=10,
            generator=deepcopy(generator),
            data_preprocessor=DataPreprocessor())

        # no mode
        inputs = dict(inputs=dict(num_batches=3))
        outputs_val = gan.val_step(inputs)
        outputs_test = gan.test_step(inputs)
        self.assertEqual(len(outputs_val), 3)
        self.assertEqual(len(outputs_test), 3)
        for idx in range(3):
            self.assertEqual(outputs_val[idx].fake_img.shape, (3, 8, 8))
            self.assertEqual(outputs_test[idx].fake_img.shape, (3, 8, 8))

        # set mode
        inputs = dict(inputs=dict(num_batches=4, sample_model='orig'))
        outputs_val = gan.val_step(inputs)
        outputs_test = gan.test_step(inputs)
        self.assertEqual(len(outputs_val), 4)
        self.assertEqual(len(outputs_test), 4)
        for idx in range(4):
            self.assertEqual(outputs_val[idx].sample_model, 'orig')
            self.assertEqual(outputs_test[idx].sample_model, 'orig')
            self.assertEqual(outputs_val[idx].fake_img.shape, (3, 8, 8))
            self.assertEqual(outputs_test[idx].fake_img.shape, (3, 8, 8))

        inputs = dict(inputs=dict(num_batches=4, sample_model='orig/ema'))
        self.assertRaises(AssertionError, gan.val_step, inputs)

        inputs = dict(inputs=dict(num_batches=4, sample_model='ema'))
        self.assertRaises(AssertionError, gan.val_step, inputs)

        # set noise input
        inputs = dict(inputs=dict(noise=torch.randn(4, 10)))
        outputs_val = gan.val_step(inputs)
        outputs_test = gan.test_step(inputs)
        self.assertEqual(len(outputs_val), 4)
        self.assertEqual(len(outputs_val), 4)
        for idx in range(4):
            test_fake_img = outputs_test[idx].fake_img
            val_fake_img = outputs_val[idx].fake_img
            assert_allclose(test_fake_img, val_fake_img)

    def test_forward(self):
        # set a gan w/o EMA
        gan = ToyGAN(
            noise_size=10,
            generator=deepcopy(generator),
            data_preprocessor=DataPreprocessor())
        inputs = dict(num_batches=3)
        outputs = gan(inputs, None)
        self.assertEqual(len(outputs), 3)
        for idx in range(3):
            self.assertTrue(outputs[idx].fake_img.shape == (3, 8, 8))

        outputs = gan(inputs)
        self.assertEqual(len(outputs), 3)
        for idx in range(3):
            self.assertEqual(outputs[idx].fake_img.shape, (3, 8, 8))

        outputs = gan(torch.randn(3, 10))
        self.assertEqual(len(outputs), 3)
        for idx in range(3):
            self.assertEqual(outputs[idx].fake_img.shape, (3, 8, 8))

        # set a gan w EMA
        gan = ToyGAN(
            noise_size=10,
            generator=deepcopy(generator),
            data_preprocessor=DataPreprocessor(),
            ema_config=dict(interval=1))
        inputs = dict(num_batches=3)
        outputs = gan(inputs)
        self.assertEqual(len(outputs), 3)
        for idx in range(3):
            self.assertEqual(outputs[idx].fake_img.shape, (3, 8, 8))

        inputs = dict(num_batches=3, sample_model='ema/orig')
        outputs = gan(inputs)
        self.assertEqual(len(outputs), 3)
        for idx in range(3):
            ema_img = outputs[idx].ema
            orig_img = outputs[idx].orig
            self.assertEqual(ema_img.fake_img.shape, orig_img.fake_img.shape)
            self.assertTrue(outputs[idx].sample_model, 'ema/orig')

        inputs = dict(noise=torch.randn(4, 10))
        outputs = gan(inputs)
        self.assertEqual(len(outputs), 4)
        for idx in range(4):
            self.assertEqual(outputs[idx].fake_img.shape, (3, 8, 8))

        # test additional sample kwargs
        sample_kwargs = dict(return_noise=True)
        inputs = dict(noise=torch.randn(4, 10), sample_kwargs=sample_kwargs)
        outputs = gan(inputs)
        self.assertEqual(len(outputs), 4)
        for idx in range(4):
            self.assertEqual(outputs[idx].fake_img.shape, (3, 8, 8))

        # test when data samples is not None
        inputs = dict(num_batches=3, sample_model='ema/orig')
        data_samples = [DataSample(id=1), DataSample(id=2), DataSample(id=3)]
        outputs = gan(inputs, DataSample.stack(data_samples))
        self.assertEqual(len(outputs), 3)
        for idx, output in enumerate(outputs):
            self.assertEqual(output.id, idx + 1)

    def test_custom_loss(self):
        message_hub = MessageHub.get_instance('basegan-test-custom-loss')
        message_hub.update_info('iter', 10)

        gan_loss = dict(type='GANLossComps', gan_type='vanilla')

        # test loss config is dict()
        gan = BaseGAN(
            noise_size=5,
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
        gan = BaseGAN(
            noise_size=5,
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

        gan = BaseGAN(
            noise_size=5,
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
        gan = BaseGAN(
            noise_size=5,
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

        data = dict(inputs=dict(img=torch.randn(2, 3, 32, 32)))
        log_vars = gan.train_step(data, optim_wrapper=optimizer_wrapper)
        self.assertIn('loss', log_vars)
        self.assertIn('loss_disc_fake', log_vars)
        self.assertIn('loss_disc_real', log_vars)
        self.assertIn('loss_disc_fake_g', log_vars)
        self.assertIn('loss_gen_aux', log_vars)
        self.assertIn('loss_disc_shift', log_vars)

        # test forward with only gan loss
        loss_config = dict(gan_loss=gan_loss)
        gan = BaseGAN(
            noise_size=5,
            generator=deepcopy(generator),
            discriminator=deepcopy(discriminator),
            data_preprocessor=DataPreprocessor(),
            loss_config=loss_config)
        data = dict(inputs=dict(img=torch.randn(2, 3, 32, 32)))
        log_vars = gan.train_step(data, optim_wrapper=optimizer_wrapper)
        self.assertIn('loss', log_vars)
        self.assertIn('loss_disc_fake', log_vars)
        self.assertIn('loss_disc_real', log_vars)
        self.assertIn('loss_disc_fake_g', log_vars)
        self.assertNotIn('loss_gen_aux', log_vars)
        self.assertNotIn('loss_disc_shift', log_vars)

    def test_gather_log_vars(self):
        gan = ToyGAN(
            noise_size=5,
            generator=deepcopy(generator),
            discriminator=deepcopy(discriminator),
            data_preprocessor=DataPreprocessor())
        log_dict_list = [
            dict(loss=torch.Tensor([2.33]), loss_disc=torch.Tensor([1.14514]))
        ]
        self.assertDictEqual(log_dict_list[0],
                             gan.gather_log_vars(log_dict_list))

        log_dict_list = [
            dict(loss=torch.Tensor([2]), loss_disc=torch.Tensor([2])),
            dict(loss=torch.Tensor([3]), loss_disc=torch.Tensor([5]))
        ]
        self.assertDictEqual(
            dict(loss=torch.Tensor([2.5]), loss_disc=torch.Tensor([3.5])),
            gan.gather_log_vars(log_dict_list))

        # test raise error
        with self.assertRaises(AssertionError):
            gan.gather_log_vars([dict(a=1), dict(b=2)])


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
