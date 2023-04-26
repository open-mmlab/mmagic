# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple, Union

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from mmengine import Config
from torch import Tensor

from mmagic.registry import MODELS
from ..pggan import ProgressiveGrowingGAN

ModelType = Union[Dict, nn.Module]
TrainInput = Union[dict, Tensor]


@MODELS.register_module('StyleGANV1')
@MODELS.register_module('StyleGANv1')
@MODELS.register_module()
class StyleGAN1(ProgressiveGrowingGAN):
    """Implementation of `A Style-Based Generator Architecture for Generative
    Adversarial Networks`.

    <https://openaccess.thecvf.com/content_CVPR_2019/html/Karras_A_Style-Based_Generator_Architecture_for_Generative_Adversarial_Networks_CVPR_2019_paper.html>`_  # noqa
    (StyleGANv1). This class is inherited from
    :class:`~ProgressiveGrowingGAN` to support progressive training.

    Detailed architecture can be found in
    :class:`~mmagic.models.editors.stylegan1.StyleGAN1Generator`
    and
    :class:`~mmagic.models.editors.stylegan1.StyleGAN1Discriminator`

    Args:
        generator (ModelType): The config or model of the generator.
        discriminator (Optional[ModelType]): The config or model of the
            discriminator. Defaults to None.
        data_preprocessor (Optional[Union[dict, Config]]): The pre-process
            config or :class:`~mmagic.models.DataPreprocessor`.
        style_channels (int): The number of channels for style code. Defaults
            to 128.
        nkimgs_per_scale (dict): The number of images need for each
            resolution's training. Defaults to `{}`.
        intep_real (dict, optional): The config of interpolation method for
            real images. If not passed, bilinear interpolation with
            align_corners will be used. Defaults to None.
        transition_kimgs (int, optional): The number of images during used to
            transit from the previous torgb layer to newer torgb layer.
            Defaults to 600.
        prev_stage (int, optional): The resolution of previous stage. Used for
            resume training. Defaults to 0.
        ema_config (Optional[Dict]): The config for generator's exponential
            moving average setting. Defaults to None.
    """

    def __init__(self,
                 generator: ModelType,
                 discriminator: Optional[ModelType] = None,
                 data_preprocessor: Optional[Union[dict, Config]] = None,
                 style_channels: int = 512,
                 nkimgs_per_scale: dict = {},
                 interp_real: Optional[dict] = None,
                 transition_kimgs: int = 600,
                 prev_stage: int = 0,
                 ema_config: Optional[Dict] = None):

        # get valid style_channels
        if isinstance(generator, dict):
            model_style_channels = generator.get('style_channels', None)
        else:
            model_style_channels = getattr(generator, 'style_channels', None)

        if style_channels is not None and model_style_channels is not None:
            assert style_channels == model_style_channels, (
                'Input \'style_channels\' is inconsistent with '
                f'\'generator.style_channels\'. Receive \'{style_channels}\' '
                f'and \'{model_style_channels}\'.')
        else:
            style_channels = style_channels or model_style_channels

        super().__init__(generator, discriminator, data_preprocessor,
                         nkimgs_per_scale, None, interp_real, transition_kimgs,
                         prev_stage, ema_config)

        self.noise_size = style_channels

    def disc_loss(self, disc_pred_fake: Tensor, disc_pred_real: Tensor,
                  fake_data: Tensor, real_data: Tensor) -> Tuple[Tensor, dict]:
        r"""Get disc loss. StyleGANv1 use non-saturating gan loss and R1
        gradient penalty. loss to train the discriminator.

        .. math:
            L_{D} = \mathbb{E}_{z\sim{p_{z}}}D\left\(G\left\(z\right\)\right\)
                - \mathbb{E}_{x\sim{p_{data}}}D\left\(x\right\) + L_{GP} \\
            L_{GP} = \lambda\mathbb{E}(\Vert\nabla_{\tilde{x}}D(\tilde{x})
                \Vert_2-1)^2 \\
            \tilde{x} = \epsilon x + (1-\epsilon)G(z)
            L_{shift} =

        Args:
            disc_pred_fake (Tensor): Discriminator's prediction of the fake
                images.
            disc_pred_real (Tensor): Discriminator's prediction of the real
                images.
            fake_data (Tensor): Generated images, used to calculate gradient
                penalty.
            real_data (Tensor): Real images, used to calculate gradient
                penalty.

        Returns:
            Tuple[Tensor, dict]: Loss value and a dict of log variables.
        """

        losses_dict = dict()
        losses_dict['loss_disc_fake'] = F.softplus(disc_pred_fake).mean()
        losses_dict['loss_disc_real'] = F.softplus(-disc_pred_real).mean()

        # R1 gradient penalty
        batch_size = real_data.size(0)
        real_data_ = real_data.clone().requires_grad_()
        disc_pred = self.discriminator(
            real_data_,
            curr_scale=self.curr_scale[0],
            transition_weight=self._curr_transition_weight)
        gradients = autograd.grad(
            outputs=disc_pred,
            inputs=real_data_,
            grad_outputs=torch.ones_like(disc_pred),
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        # norm_mode is 'HWC'
        gradients_penalty = gradients.pow(2).reshape(batch_size,
                                                     -1).sum(1).mean()
        losses_dict['loss_r1_gp'] = 10 * gradients_penalty

        parsed_loss, log_vars = self.parse_losses(losses_dict)
        return parsed_loss, log_vars

    def gen_loss(self, disc_pred_fake: Tensor) -> Tuple[Tensor, dict]:
        r"""Generator loss for PGGAN. PGGAN use WGAN's loss to train the
        generator.

        .. math:
            L_{G} = -\mathbb{E}_{z\sim{p_{z}}}D\left\(G\left\(z\right\)\right\)
                + L_{MSE}

        Args:
            disc_pred_fake (Tensor): Discriminator's prediction of the fake
                images.

        Returns:
            Tuple[Tensor, dict]: Loss value and a dict of log variables.
        """
        losses_dict = dict()
        losses_dict['loss_gen'] = -disc_pred_fake.mean()
        loss, log_vars = self.parse_losses(losses_dict)
        return loss, log_vars
