# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial

import pytest
import torch

from mmagic.models.editors.dcgan import DCGANDiscriminator
from mmagic.models.editors.pggan import PGGANDiscriminator
from mmagic.models.losses import (DiscShiftLossComps, GradientPenaltyLossComps,
                                  R1GradientPenaltyComps)
from mmagic.models.losses.gan_loss import (gradient_penalty_loss,
                                           r1_gradient_penalty_loss)


class TestDiscShiftLoss(object):

    @classmethod
    def setup_class(cls):
        cls.input_tensor = torch.randn((2, 10))
        cls.default_cfg = dict(
            loss_weight=0.1, data_info=dict(pred='disc_pred'))
        cls.default_input_dict = dict(disc_pred=cls.input_tensor)

    def test_module_wrapper(self):
        # test with default config
        loss_module = DiscShiftLossComps(**self.default_cfg)
        loss = loss_module(self.default_input_dict)
        assert loss.ndim == 2

        with pytest.raises(NotImplementedError):
            _ = loss_module(self.default_input_dict, 1)

        with pytest.raises(AssertionError):
            _ = loss_module(1, outputs_dict=self.default_input_dict)
        input_ = dict(outputs_dict=self.default_input_dict)
        loss = loss_module(**input_)
        assert loss.ndim == 2

        with pytest.raises(AssertionError):
            _ = loss_module(self.input_tensor)

        # test without data_info
        loss_module = DiscShiftLossComps(data_info=None)
        loss = loss_module(self.input_tensor)
        assert loss.ndim == 2


class TestGradientPenalty:

    @classmethod
    def setup_class(cls):
        cls.input_img = torch.randn((2, 3, 8, 8))
        cls.disc = DCGANDiscriminator(
            input_scale=8, output_scale=4, out_channels=5)
        cls.pggan_disc = PGGANDiscriminator(
            in_scale=8, base_channels=32, max_channels=32)
        cls.data_info = dict(
            discriminator='disc', real_data='real_imgs', fake_data='fake_imgs')

    def test_gp_loss(self):
        loss = gradient_penalty_loss(self.disc, self.input_img, self.input_img)
        assert loss > 0

        loss = gradient_penalty_loss(
            self.disc, self.input_img, self.input_img, norm_mode='HWC')
        assert loss > 0

        with pytest.raises(NotImplementedError):
            _ = gradient_penalty_loss(
                self.disc, self.input_img, self.input_img, norm_mode='xxx')

        loss = gradient_penalty_loss(
            self.disc, self.input_img, self.input_img, norm_mode='HWC')
        assert loss > 0

        loss = gradient_penalty_loss(
            self.disc,
            self.input_img,
            self.input_img,
            norm_mode='HWC',
            mask=torch.ones_like(self.input_img))
        assert loss > 0

        data_dict = dict(
            real_imgs=self.input_img,
            fake_imgs=self.input_img,
            disc=partial(self.pggan_disc, transition_weight=0.5, curr_scale=8))
        gp_loss = GradientPenaltyLossComps(
            loss_weight=10, norm_mode='pixel', data_info=self.data_info)

        loss = gp_loss(data_dict)
        assert loss > 0
        loss = gp_loss(outputs_dict=data_dict)
        assert loss > 0
        with pytest.raises(NotImplementedError):
            _ = gp_loss(asdf=1.)

        with pytest.raises(AssertionError):
            _ = gp_loss(1.)

        with pytest.raises(AssertionError):
            _ = gp_loss(1., 2, outputs_dict=data_dict)


class TestR1GradientPenalty:

    @classmethod
    def setup_class(cls):
        cls.data_info = dict(discriminator='disc', real_data='real_imgs')
        cls.disc = DCGANDiscriminator(
            input_scale=8, output_scale=4, out_channels=5)
        cls.pggan_disc = PGGANDiscriminator(
            in_scale=8, base_channels=32, max_channels=32)
        cls.input_img = torch.randn((2, 3, 8, 8))

    def test_r1_regularizer(self):
        loss = r1_gradient_penalty_loss(self.disc, self.input_img)
        assert loss > 0

        loss = r1_gradient_penalty_loss(
            self.disc, self.input_img, norm_mode='HWC')
        assert loss > 0

        with pytest.raises(NotImplementedError):
            _ = r1_gradient_penalty_loss(
                self.disc, self.input_img, norm_mode='xxx')

        loss = r1_gradient_penalty_loss(
            self.disc, self.input_img, norm_mode='HWC')
        assert loss > 0

        loss = r1_gradient_penalty_loss(
            self.disc,
            self.input_img,
            norm_mode='HWC',
            mask=torch.ones_like(self.input_img))
        assert loss > 0

        data_dict = dict(
            real_imgs=self.input_img,
            disc=partial(self.pggan_disc, transition_weight=0.5, curr_scale=8))
        gp_loss = R1GradientPenaltyComps(
            loss_weight=10, norm_mode='pixel', data_info=self.data_info)

        loss = gp_loss(data_dict)
        assert loss > 0
        loss = gp_loss(outputs_dict=data_dict)
        assert loss > 0
        with pytest.raises(NotImplementedError):
            _ = gp_loss(asdf=1.)

        with pytest.raises(AssertionError):
            _ = gp_loss(1.)

        with pytest.raises(AssertionError):
            _ = gp_loss(1., 2, outputs_dict=data_dict)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
