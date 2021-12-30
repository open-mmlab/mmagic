# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import auto_fp16

from ..builder import build_loss
from ..registry import MODELS
from .base_mattor import BaseMattor
from .utils import get_unknown_tensor


@MODELS.register_module()
class IndexNet(BaseMattor):
    """IndexNet matting model.

    This implementation follows:
    Indices Matter: Learning to Index for Deep Image Matting

    Args:
        backbone (dict): Config of backbone.
        train_cfg (dict): Config of training. In 'train_cfg', 'train_backbone'
            should be specified.
        test_cfg (dict): Config of testing.
        pretrained (str): path of pretrained model.
        loss_alpha (dict): Config of the alpha prediction loss. Default: None.
        loss_comp (dict): Config of the composition loss. Default: None.
    """

    def __init__(self,
                 backbone,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 loss_alpha=None,
                 loss_comp=None):
        super().__init__(backbone, None, train_cfg, test_cfg, pretrained)

        self.loss_alpha = (
            build_loss(loss_alpha) if loss_alpha is not None else None)
        self.loss_comp = (
            build_loss(loss_comp) if loss_comp is not None else None)

        # support fp16
        self.fp16_enabled = False

    def forward_dummy(self, inputs):
        return self.backbone(inputs)

    @auto_fp16(apply_to=('merged', 'trimap'))
    def forward_train(self, merged, trimap, meta, alpha, ori_merged, fg, bg):
        """Forward function for training IndexNet model.

        Args:
            merged (Tensor): Input images tensor with shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            trimap (Tensor): Tensor of trimap with shape (N, 1, H, W).
            meta (list[dict]): Meta data about the current data batch.
            alpha (Tensor): Tensor of alpha with shape (N, 1, H, W).
            ori_merged (Tensor): Tensor of origin merged images (not
                normalized) with shape (N, C, H, W).
            fg (Tensor): Tensor of foreground with shape (N, C, H, W).
            bg (Tensor): Tensor of background with shape (N, C, H, W).

        Returns:
            dict: Contains the loss items and batch information.
        """
        pred_alpha = self.backbone(torch.cat((merged, trimap), 1))

        losses = dict()
        weight = get_unknown_tensor(trimap, meta)
        if self.loss_alpha is not None:
            losses['loss_alpha'] = self.loss_alpha(pred_alpha, alpha, weight)
        if self.loss_comp is not None:
            losses['loss_comp'] = self.loss_comp(pred_alpha, fg, bg,
                                                 ori_merged, weight)
        return {'losses': losses, 'num_samples': merged.size(0)}

    def forward_test(self,
                     merged,
                     trimap,
                     meta,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        """Defines the computation performed at every test call.

        Args:
            merged (Tensor): Image to predict alpha matte.
            trimap (Tensor): Trimap of the input image.
            meta (list[dict]): Meta data about the current data batch.
                Currently only batch_size 1 is supported. It may contain
                information needed to calculate metrics (``ori_alpha`` and
                ``ori_trimap``) or save predicted alpha matte
                (``merged_path``).
            save_image (bool, optional): Whether save predicted alpha matte.
                Defaults to False.
            save_path (str, optional): The directory to save predicted alpha
                matte. Defaults to None.
            iteration (int, optional): If given as None, the saved alpha matte
                will have the same file name with ``merged_path`` in meta dict.
                If given as an int, the saved alpha matte would named with
                postfix ``_{iteration}.png``. Defaults to None.

        Returns:
            dict: Contains the predicted alpha and evaluation result.
        """
        pred_alpha = self.backbone(torch.cat((merged, trimap), 1))

        pred_alpha = pred_alpha.cpu().numpy().squeeze()
        pred_alpha = self.restore_shape(pred_alpha, meta)
        eval_result = self.evaluate(pred_alpha, meta)

        if save_image:
            self.save_image(pred_alpha, meta, save_path, iteration)

        return {'pred_alpha': pred_alpha, 'eval_result': eval_result}
