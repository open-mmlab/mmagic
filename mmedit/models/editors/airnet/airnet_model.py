# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmengine import MessageHub

from mmedit.models.base_models import BaseEditModel
from mmedit.registry import MODELS


@MODELS.register_module()
class AirNetRestorer(BaseEditModel):
    """AirNet restorer model for single image restoration for unknown tasks.

    Ref: "All-In-One Image Restoration for Unknown Corruption"

    Note: This class mainly handle the problem that:
        AirNet training will contain two losses:
            reconstruction loss and contrastive loss

    Args:
        generator (dict): Config for the generator.
        pixel_loss (dict): Config for the pixel loss.
        pretrained (str): Path for pretrained model. Default: None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
    """

    def __init__(self,
                 generator,
                 pixel_loss,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 data_preprocessor=None,
                 train_patch_size=128):
        super().__init__(generator, pixel_loss, train_cfg, test_cfg, init_cfg,
                         data_preprocessor)
        self.train_patch_size = train_patch_size
        self.ce_loss = nn.CrossEntropyLoss()

    def forward_tensor(self, inputs, data_samples=None, **kwargs):
        """Forward tensor. Returns result of simple forward.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.

        Returns:
            Tensor: result of simple forward.
        """
        outputs = self.generator(inputs, **kwargs)
        return outputs['restored']

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
        message_hub = MessageHub.get_current_instance()
        curr_epoch = message_hub.get_info('epoch')
        # crop two patches for training
        # only restore patch 1
        # patch 2 only used for contrastive learning
        inputs_dict = self._double_crop(inputs, data_samples)
        outputs = self.generator(inputs_dict)

        logits = outputs['logits']
        labels = outputs['labels']
        contrastive_loss = self.ce_loss(logits, labels)
        loss_dict = dict(
            loss=contrastive_loss, contrastive_loss=contrastive_loss)

        if curr_epoch >= self.train_cfg['epochs_encoder']:
            # For training epoch greater than `epochs_encoder`,
            # train encoder and restorer
            # with contrastive loss and pixel_loss
            batch_pred_patch_1 = outputs['restored']
            batch_clear_patch_1 = inputs_dict['clear_patch_1']
            pixel_loss = self.pixel_loss(batch_pred_patch_1,
                                         batch_clear_patch_1)
            loss_dict['loss'] = pixel_loss + 0.1 * contrastive_loss
            loss_dict['pixel_loss'] = pixel_loss

        return loss_dict

    def _double_crop(self, inputs, data_samples):
        """AirNet extra data preparation.

        For contrastive learning in AirNet,
        two cropped patches in one pair will be used,
        because the degradation in one image is assumed to be identical.

        Args:
        inputs (torch.Tensor): batch input tensor collated by
            :attr:`data_preprocessor`.
        data_samples (List[BaseDataElement], optional):
            data samples collated by :attr:`data_preprocessor`.

        Returns:
            dict: degrad_patch_1 and degrad_patch_2.
        """
        gt_imgs = [data_sample.gt_img.data for data_sample in data_samples]
        batch_gt_data = torch.stack(gt_imgs)

        degrad_patch_1, clear_patch_1 = self._crop_patch(inputs, batch_gt_data)
        degrad_patch_2, clear_patch_2 = self._crop_patch(inputs, batch_gt_data)
        inputs_dict = dict(
            degrad_patch_1=degrad_patch_1,
            degrad_patch_2=degrad_patch_2,
            clear_patch_1=clear_patch_1,
            clear_patch_2=clear_patch_2)

        return inputs_dict

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[-2]
        W = img_1.shape[-1]
        assert (H >= self.train_patch_size) and (W >= self.train_patch_size), \
            'Input image size is smaller than the train patch size'
        ind_H = torch.randint(0, H - self.train_patch_size, (1, 1))
        ind_W = torch.randint(0, W - self.train_patch_size, (1, 1))

        patch_1 = img_1[..., ind_H:ind_H + self.train_patch_size,
                        ind_W:ind_W + self.train_patch_size]
        patch_2 = img_2[..., ind_H:ind_H + self.train_patch_size,
                        ind_W:ind_W + self.train_patch_size]

        return patch_1, patch_2
