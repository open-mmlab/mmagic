import os

import torch
from mmcv.runner import auto_fp16

from ..builder import build_loss
from ..registry import MODELS
from .base_mattor import BaseMattor
from .utils import fba_fusion


@MODELS.register_module()
class FBA(BaseMattor):
    """F, B, Alpha Matting.

    https://arxiv.org/pdf/2003.07711.pdf

    Args:
        backbone (dict): Config of backbone.
        train_cfg (dict): Config of training. In ``train_cfg``,
            ``train_backbone`` should be specified. If the model has a refiner,
            ``train_refiner`` should be specified.
        test_cfg (dict): Config of testing. In ``test_cfg``, If the model has a
            refiner, ``train_refiner`` should be specified.
        pretrained (str): Path of the pretrained model.
        loss_* (dict): Config of the alpha prediction loss. Default: None.
    """

    def __init__(self,
                 backbone,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 loss_alpha=None,
                 loss_alpha_compo=None,
                 loss_alpha_lap=None,
                 loss_alpha_grad=None,
                 loss_fb=None,
                 loss_fb_compo=None,
                 loss_fb_lap=None,
                 loss_exclusion=None):
        super(FBA, self).__init__(
            backbone,
            None,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        self.loss_alpha = build_loss(
            loss_alpha) if loss_alpha is not None else None
        self.loss_alpha_compo = build_loss(
            loss_alpha_compo) if loss_alpha_compo is not None else None
        self.loss_alpha_lap = build_loss(
            loss_alpha_lap) if loss_alpha_lap is not None else None
        self.loss_alpha_grad = build_loss(
            loss_alpha_grad) if loss_alpha_grad is not None else None

        self.loss_fb = build_loss(loss_fb) if loss_fb is not None else None
        self.loss_fb_compo = build_loss(
            loss_fb_compo) if loss_fb_compo is not None else None
        self.loss_exclusion = build_loss(
            loss_exclusion) if loss_exclusion is not None else None
        self.loss_fb_lap = build_loss(
            loss_fb_lap) if loss_fb_lap is not None else None
        # support fp16
        self.fp16_enabled = False

    def forward(self,
                merged,
                trimap_transformed,
                two_channel_trimap,
                merged_unnormalized,
                meta,
                fg=None,
                bg=None,
                alpha=None,
                test_mode=False,
                **kwargs):

        if not test_mode:
            return self.forward_train(merged, trimap_transformed,
                                      two_channel_trimap, merged_unnormalized,
                                      alpha, fg, bg, **kwargs)
        else:
            return self.forward_test(merged, trimap_transformed,
                                     two_channel_trimap, merged_unnormalized,
                                     meta, **kwargs)

    @auto_fp16(apply_to=('x', ))
    def _forward(self, x):
        output_fba = self.backbone(x)
        return output_fba

    def forward_dummy(self, inputs):
        return self._forward(inputs)

    def forward_train(self, merged, trimap_transformed, two_channel_trimap,
                      merged_unnormalized, alpha, fg, bg, ori_fg):
        """Forward function for training FBA model.

        Args:
            merged (Tensor): with shape (N, C, H, W) encoded
                input images. Typically these should be mean centered and
                std scaled.
            trimap_transformed (Tensor): with shape (N, 6, H, W).
                Tensor of trimap.
            two_channel_trimap (Tensor): with shape (N, 2, H, W).
                Tensor of trimap.
            merged_unnormalized (Tensor): with shape (N, C, H, W).
                Tensor of unnormalized merged image.
            alpha (Tensor): with shape (N, 1, H, W). Tensor of alpha.
            fg (Tensor): with shape (N, 3, H, W). Tensor of fg.
                Fg extended to the whole image.
            bg (Tensor): with shape (N, 3, H, W). Tensor of bg.
            ori_fg (Tensor): with shape (N, 3, H, W). Tensor of ori_fg.

        Returns:
            dict: Contains the loss items and batch infomation.
        """

        input = torch.cat((merged, trimap_transformed, two_channel_trimap,
                           merged_unnormalized), 1)
        pred_alpha, pred_fg, pred_bg = self._forward(input)
        threshold = 0.01
        mask = alpha > threshold
        nmask = ~mask
        losses = dict()
        if self.loss_alpha is not None:
            losses['loss_alpha'] = self.loss_alpha(pred_alpha, alpha)
        if self.loss_alpha_compo is not None:
            losses['loss_alpha_compo'] = self.loss_alpha_compo(
                pred_alpha,
                ori_fg,
                bg,
                merged_unnormalized,
                alpha=alpha,
                threshold=0.01)
        if self.loss_alpha_grad is not None:
            losses['loss_alpha_grad'] = self.loss_alpha_grad(pred_alpha, alpha)
        if self.loss_alpha_lap is not None:
            losses['loss_alpha_lap'] = self.loss_alpha_lap(alpha, pred_alpha)

        if self.loss_fb is not None:
            losses['loss_fb'] = self.loss_fb(
                pred_fg * nmask, ori_fg * nmask) + self.loss_fb(
                    pred_bg * mask, bg * mask) + self.loss_fb(
                        pred_fg, ori_fg) + self.loss_fb(pred_bg, bg)
        if self.loss_fb_compo is not None:
            losses['loss_fb_compo'] = self.loss_fb_compo(
                pred_fg, pred_bg, alpha, merged_unnormalized)
        if self.loss_exclusion is not None:
            losses['loss_exclusion'] = self.loss_exclusion(pred_bg, pred_fg)
        if self.loss_fb_lap is not None:
            losses['loss_fb_lap'] = self.loss_fb_lap(
                fg, pred_fg) + self.loss_fb_lap(bg, pred_bg)

        return {'losses': losses, 'num_samples': merged.size(0)}

    def forward_test(self,
                     merged,
                     trimap_transformed,
                     two_channel_trimap,
                     merged_unnormalized,
                     meta,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        """Defines the computation performed at every test call.

        Args:
            merged (Tensor): with shape (N, C, H, W) encoded
                input images. Typically these should be mean centered and
                std scaled.
            trimap_transformed (Tensor): with shape (N, 6, H, W).
                Tensor of trimap.
            two_channel_trimap (Tensor): with shape (N, 2, H, W).
                Tensor of trimap.
            merged_unnormalized (Tensor): with shape (N, C, H, W).
                Tensor of unnormalized merged image.
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
        # for batch size 1
        input = torch.cat((merged, trimap_transformed, two_channel_trimap,
                           merged_unnormalized), 1)
        pred_alpha, pred_fg, pred_bg = self._forward(input)
        # FBA Fusion
        pred_alpha, pred_fg, pred_bg = fba_fusion(pred_alpha,
                                                  merged_unnormalized, pred_fg,
                                                  pred_bg)
        pred_alpha = pred_alpha.detach().cpu().numpy().squeeze()

        pred_alpha = self.restore_shape(pred_alpha, meta)
        pred_fg = pred_fg.detach().cpu().numpy().squeeze().transpose(
            (1, 2, 0))[:, :, ::-1]
        pred_fg = self.restore_shape(pred_fg, meta)
        pred_bg = pred_bg.detach().cpu().numpy().squeeze().transpose(
            (1, 2, 0))[:, :, ::-1]
        pred_bg = self.restore_shape(pred_bg, meta)
        trimap = meta[0]['ori_trimap']
        ori_merged = meta[0]['ori_merged']
        pred_fg[trimap == 255] = ori_merged[trimap == 255]
        pred_bg[trimap == 0] = ori_merged[trimap == 0]
        eval_result = self.evaluate(pred_alpha, meta)
        if save_image:
            save_a = os.path.join(save_path, 'alpha')
            save_f = os.path.join(save_path, 'fg')
            save_b = os.path.join(save_path, 'bg')

            self.save_image(pred_alpha, meta, save_a, iteration)
            self.save_image(pred_fg, meta, save_f, iteration)
            self.save_image(pred_bg, meta, save_b, iteration)

        return {'pred_alpha': pred_alpha, 'eval_result': eval_result}
