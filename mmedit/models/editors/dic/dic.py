# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.registry import MODELS
from ..srgan import SRGAN


@MODELS.register_module()
class DIC(SRGAN):
    """DIC model for Face Super-Resolution.

    Paper: Deep Face Super-Resolution with Iterative Collaboration between
        Attentive Recovery and Landmark Estimation.

    Args:
        generator (dict): Config for the generator.
        pixel_loss (dict): Config for the pixel loss.
        align_loss (dict): Config for the align loss.
        discriminator (dict): Config for the discriminator. Default: None.
        gan_loss (dict): Config for the gan loss. Default: None.
        feature_loss (dict): Config for the feature loss. Default: None.
        train_cfg (dict): Config for train. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`. Default: None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`. Default: None.
    """

    def __init__(self,
                 generator,
                 pixel_loss,
                 align_loss,
                 discriminator=None,
                 gan_loss=None,
                 feature_loss=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 data_preprocessor=None):

        super().__init__(
            generator=generator,
            discriminator=discriminator,
            gan_loss=gan_loss,
            pixel_loss=pixel_loss,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)

        self.align_loss = MODELS.build(align_loss)
        self.feature_loss = MODELS.build(
            feature_loss) if feature_loss else None

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

        sr_list, heatmap_list = self.generator(inputs)

        if training:
            return sr_list, heatmap_list
        else:
            return sr_list[-1]

    def if_run_g(self):
        """Calculates whether need to run the generator step."""

        return True

    def if_run_d(self):
        """Calculates whether need to run the discriminator step."""

        return self.step_counter >= self.pixel_init and super().if_run_d()

    def g_step(self, batch_outputs, batch_gt_data):
        """G step of GAN: Calculate losses of generator.

        Args:
            batch_outputs (Tensor): Batch output of generator.
            batch_gt_data (Tensor): Batch GT data.

        Returns:
            dict: Dict of losses.
        """

        sr_list, heatmap_list = batch_outputs
        gt, gt_heatmap = batch_gt_data

        losses = dict()

        # pix loss
        for step, (sr, heatmap) in enumerate(zip(sr_list, heatmap_list)):
            losses[f'loss_pixel_v{step}'] = self.pixel_loss(sr, gt)
            losses[f'loss_align_v{step}'] = self.align_loss(
                heatmap, gt_heatmap)

        if self.step_counter >= self.pixel_init:
            pred = sr_list[-1]

            # perceptual loss
            if self.feature_loss:
                loss_feature = self.feature_loss(pred, gt)
                losses['loss_feature'] = loss_feature

            # gan loss for generator
            if self.gan_loss and self.discriminator:
                fake_g_pred = self.discriminator(pred)
                losses['loss_gan'] = self.gan_loss(
                    fake_g_pred, target_is_real=True, is_disc=False)

        return losses

    def d_step_with_optim(self, batch_outputs, batch_gt_data, optim_wrapper):
        """D step with optim of GAN: Calculate losses of discriminator and run
        optim.

        Args:
            batch_outputs (Tuple[Tensor]): Batch output of generator.
            batch_gt_data (Tuple[Tensor]): Batch GT data.
            optim_wrapper (OptimWrapper): Optim wrapper of discriminator.

        Returns:
            dict: Dict of parsed losses.
        """

        sr_list, _ = batch_outputs
        gt, _ = batch_gt_data

        return super().d_step_with_optim(
            batch_outputs=sr_list[-1],
            batch_gt_data=gt,
            optim_wrapper=optim_wrapper)

    @staticmethod
    def extract_gt_data(data_samples):
        """extract gt data from data samples.

        Args:
            data_samples (list): List of EditDataSample.

        Returns:
            Tensor: Extract gt data.
        """

        gt_imgs = [data_sample.gt_img.data for data_sample in data_samples]
        batch_gt_img = torch.stack(gt_imgs)
        gt_heatmaps = [
            data_sample.gt_heatmap.data for data_sample in data_samples
        ]
        batch_gt_heatmap = torch.stack(gt_heatmaps)

        return [batch_gt_img, batch_gt_heatmap]
