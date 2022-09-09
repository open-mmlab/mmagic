# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.optim import OptimWrapperDict

from mmedit.models.utils import set_requires_grad
from mmedit.registry import MODELS
from mmedit.structures import EditDataSample
from ..srgan import SRGAN


@MODELS.register_module()
class TTSR(SRGAN):
    """TTSR model for Reference-based Image Super-Resolution.

    Paper: Learning Texture Transformer Network for Image Super-Resolution.

    Args:
        generator (dict): Config for the generator.
        extractor (dict): Config for the extractor.
        transformer (dict): Config for the transformer.
        pixel_loss (dict): Config for the pixel loss.
        discriminator (dict): Config for the discriminator. Default: None.
        perceptual_loss (dict): Config for the perceptual loss. Default: None.
        transferal_perceptual_loss (dict): Config for the transferal perceptual
            loss. Default: None.
        gan_loss (dict): Config for the GAN loss. Default: None
        train_cfg (dict): Config for train. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`. Default: None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`. Default: None.
    """

    def __init__(self,
                 generator,
                 extractor,
                 transformer,
                 pixel_loss,
                 discriminator=None,
                 perceptual_loss=None,
                 transferal_perceptual_loss=None,
                 gan_loss=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 data_preprocessor=None):

        super().__init__(
            generator=generator,
            discriminator=discriminator,
            gan_loss=gan_loss,
            pixel_loss=pixel_loss,
            perceptual_loss=perceptual_loss,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)

        self.transformer = MODELS.build(transformer)
        self.extractor = MODELS.build(extractor)
        extractor['requires_grad'] = False
        self.extractor_copy = MODELS.build(extractor)
        set_requires_grad(self.extractor_copy, False)

        if transferal_perceptual_loss:
            self.transferal_perceptual_loss = MODELS.build(
                transferal_perceptual_loss)
        else:
            self.transferal_perceptual_loss = None

        self.pixel_init = train_cfg.get('pixel_init', 0) if train_cfg else 0

    def forward_tensor(self, inputs, data_samples=None, training=False):
        """Forward tensor. Returns result of simple forward.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.
            training (bool): Whether is training. Default: False.

        Returns:
            (Tensor | Tuple[List[Tensor]]): results of forward inference and
                forward train.
        """

        img_lq = []
        ref_lq = []
        ref = []

        for data_sample in data_samples:
            img_lq.append(data_sample.img_lq.data / 255.)
            ref_lq.append(data_sample.ref_lq.data / 255.)
            ref.append(data_sample.ref_img.data / 255.)

        img_lq = torch.stack(img_lq)
        ref_lq = torch.stack(ref_lq)
        ref = torch.stack(ref)

        img_lq, _, _ = self.extractor(img_lq)
        ref_lq, _, _ = self.extractor(ref_lq)
        refs = self.extractor(ref)

        soft_attention, textures = self.transformer(img_lq, ref_lq, refs)

        pred = self.generator(inputs, soft_attention, textures)

        if training:
            return pred, soft_attention, textures
        else:
            return pred

    def if_run_g(self):
        """Calculates whether need to run the generator step."""

        return True

    def if_run_d(self):
        """Calculates whether need to run the discriminator step."""

        return self.step_counter >= self.pixel_init and super().if_run_d()

    def g_step(self, batch_outputs, batch_gt_data: EditDataSample):
        """G step of GAN: Calculate losses of generator.

        Args:
            batch_outputs (Tensor): Batch output of generator.
            batch_gt_data (Tensor): Batch GT data.

        Returns:
            dict: Dict of losses.
        """

        losses = dict()
        pred, soft_attention, textures = batch_outputs

        # pix loss
        if self.pixel_loss:
            losses['loss_pix'] = self.pixel_loss(pred, batch_gt_data)

        if self.step_counter >= self.pixel_init:
            # perceptual loss
            if self.perceptual_loss:
                loss_percep, loss_style = self.perceptual_loss(
                    pred, batch_gt_data)
                if loss_percep is not None:
                    losses['loss_perceptual'] = loss_percep
                if loss_style is not None:
                    losses['loss_style'] = loss_style

            # transform loss
            if self.transferal_perceptual_loss:
                state_dict = self.extractor.module.state_dict() if hasattr(
                    self.extractor, 'module') else self.extractor.state_dict()
                self.extractor_copy.load_state_dict(state_dict)
                sr_textures = self.extractor_copy((pred + 1.) / 2.)
                losses['loss_transferal'] = self.transferal_perceptual_loss(
                    sr_textures, soft_attention, textures)

            # gan loss for generator
            if self.gan_loss and self.discriminator:
                fake_g_pred = self.discriminator(pred)
                losses['loss_gan'] = self.gan_loss(
                    fake_g_pred, target_is_real=True, is_disc=False)

        return losses

    def g_step_with_optim(self, batch_outputs: torch.Tensor,
                          batch_gt_data: torch.Tensor,
                          optim_wrapper: OptimWrapperDict):
        """G step with optim of GAN: Calculate losses of generator and run
        optim.

        Args:
            batch_outputs (Tensor): Batch output of generator.
            batch_gt_data (Tensor): Batch GT data.
            optim_wrapper (OptimWrapperDict): Optim wrapper dict.

        Returns:
            dict: Dict of parsed losses.
        """

        g_optim_wrapper = optim_wrapper['generator']
        e_optim_wrapper = optim_wrapper['extractor']

        losses_g = self.g_step(batch_outputs, batch_gt_data)
        parsed_losses_g, log_vars_g = self.parse_losses(losses_g)

        if g_optim_wrapper.should_update():
            g_optim_wrapper.backward(parsed_losses_g)
            g_optim_wrapper.step()
            g_optim_wrapper.zero_grad()

        if e_optim_wrapper.should_update():
            e_optim_wrapper.step()
            e_optim_wrapper.zero_grad()

        return log_vars_g

    def d_step_with_optim(self, batch_outputs, batch_gt_data, optim_wrapper):
        """D step with optim of GAN: Calculate losses of discriminator and run
        optim.

        Args:
            batch_outputs (Tuple[Tensor]): Batch output of generator.
            batch_gt_data (Tensor): Batch GT data.
            optim_wrapper (OptimWrapper): Optim wrapper of discriminator.

        Returns:
            dict: Dict of parsed losses.
        """

        pred, _, _ = batch_outputs

        return super().d_step_with_optim(
            batch_outputs=pred,
            batch_gt_data=batch_gt_data,
            optim_wrapper=optim_wrapper)
