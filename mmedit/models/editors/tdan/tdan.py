# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.models import BaseEditModel
from mmedit.registry import MODELS


@MODELS.register_module()
class TDAN(BaseEditModel):
    """TDAN model for video super-resolution.

    Paper:
        TDAN: Temporally-Deformable Alignment Network for Video Super-
        Resolution, CVPR, 2020

    Args:
        generator (dict): Config for the generator structure.
        pixel_loss (dict): Config for pixel-wise loss.
        lq_pixel_loss (dict): Config for pixel-wise loss for the LQ images.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
    """

    def __init__(self,
                 generator,
                 pixel_loss,
                 lq_pixel_loss,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 data_preprocessor=None):
        super().__init__(
            generator=generator,
            pixel_loss=pixel_loss,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)

        self.lq_pixel_loss = MODELS.build(lq_pixel_loss)

    def forward_train(self, inputs, data_samples=None, **kwargs):
        """Forward training. Returns dict of losses of training.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.

        Returns:
            dict: Dict of losses.
        """

        feats, aligned_img = self.forward_tensor(
            inputs, data_samples, training=True, **kwargs)
        gt_imgs = [data_sample.gt_img.data for data_sample in data_samples]
        batch_gt_data = torch.stack(gt_imgs)

        losses = dict()
        # loss on the HR image
        losses['loss_pix'] = self.pixel_loss(feats, batch_gt_data)
        # loss on the aligned LR images
        t = aligned_img.size(1)
        lq_ref = inputs[:,
                        t // 2:t // 2 + 1, :, :, :].expand(-1, t, -1, -1, -1)
        loss_pix_lq = self.lq_pixel_loss(aligned_img, lq_ref)
        losses['loss_pix_lq'] = loss_pix_lq

        return losses

    def forward_tensor(self,
                       inputs,
                       data_samples=None,
                       training=False,
                       **kwargs):
        """Forward tensor. Returns result of simple forward.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.
            training (bool): Whether is training. Default: False.

        Returns:
            (Tensor | List[Tensor]): results of forward inference and
                forward train.
        """

        outputs = self.generator(inputs, **kwargs)

        return outputs if training else outputs[0]
