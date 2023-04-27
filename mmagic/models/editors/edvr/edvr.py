# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmagic.models import BaseEditModel
from mmagic.registry import MODELS


@MODELS.register_module()
class EDVR(BaseEditModel):
    """EDVR model for video super-resolution.

    EDVR: Video Restoration with Enhanced Deformable Convolutional Networks.

    Args:
        generator (dict): Config for the generator structure.
        pixel_loss (dict): Config for pixel-wise loss.
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

        self.with_tsa = generator.get('with_tsa', False)
        self.tsa_iter = self.train_cfg.get('tsa_iter',
                                           None) if self.train_cfg else None
        self.register_buffer('step_counter', torch.tensor(0), False)

    def forward_train(self, inputs, data_samples=None):
        """Forward training. Returns dict of losses of training.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.

        Returns:
            dict: Dict of losses.
        """

        if self.step_counter == 0 and self.with_tsa:
            if self.tsa_iter is None:
                raise KeyError(
                    'In TSA mode, train_cfg must contain "tsa_iter".')
            # only train TSA module at the beginning if with TSA module
            for k, v in self.generator.named_parameters():
                if 'fusion' not in k:
                    v.requires_grad = False

        if self.with_tsa and (self.step_counter == self.tsa_iter):
            # train all the parameters
            for v in self.generator.parameters():
                v.requires_grad = True
        self.step_counter += 1

        return super().forward_train(inputs, data_samples)
