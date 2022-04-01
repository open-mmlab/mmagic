# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import auto_fp16

from ..builder import build_loss
from ..registry import MODELS
from .base_mattor import BaseMattor
from .utils import get_unknown_tensor


@MODELS.register_module()
class GCA(BaseMattor):
    """Guided Contextual Attention image matting model.

    https://arxiv.org/abs/2001.04069

    Args:
        backbone (dict): Config of backbone.
        train_cfg (dict): Config of training. In ``train_cfg``,
            ``train_backbone`` should be specified. If the model has a refiner,
            ``train_refiner`` should be specified.
        test_cfg (dict): Config of testing. In ``test_cfg``, If the model has a
            refiner, ``train_refiner`` should be specified.
        pretrained (str): Path of the pretrained model.
        loss_alpha (dict): Config of the alpha prediction loss. Default: None.
    """

    def __init__(self,
                 backbone,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 loss_alpha=None):
        super().__init__(backbone, None, train_cfg, test_cfg, pretrained)
        self.loss_alpha = build_loss(loss_alpha)
        # support fp16
        self.fp16_enabled = False

    @auto_fp16(apply_to=('x', ))
    def _forward(self, x):
        raw_alpha = self.backbone(x)
        pred_alpha = (raw_alpha.tanh() + 1.0) / 2.0
        return pred_alpha

    def forward_dummy(self, inputs):
        return self._forward(inputs)

    def forward_train(self, merged, trimap, meta, alpha):
        """Forward function for training GCA model.

        Args:
            merged (Tensor): with shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            trimap (Tensor): with shape (N, C', H, W). Tensor of trimap. C'
                might be 1 or 3.
            meta (list[dict]): Meta data about the current data batch.
            alpha (Tensor): with shape (N, 1, H, W). Tensor of alpha.

        Returns:
            dict: Contains the loss items and batch information.
        """
        pred_alpha = self._forward(torch.cat((merged, trimap), 1))

        weight = get_unknown_tensor(trimap, meta)
        losses = {'loss': self.loss_alpha(pred_alpha, alpha, weight)}
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
        pred_alpha = self._forward(torch.cat((merged, trimap), 1))
        pred_alpha = pred_alpha.detach().cpu().numpy().squeeze()
        pred_alpha = self.restore_shape(pred_alpha, meta)
        eval_result = self.evaluate(pred_alpha, meta)

        if save_image:
            self.save_image(pred_alpha, meta, save_path, iteration)

        return {'pred_alpha': pred_alpha, 'eval_result': eval_result}
