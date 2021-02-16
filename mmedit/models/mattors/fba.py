import os

import cv2
import torch
from mmcv.runner import auto_fp16

from ..builder import build_loss
from ..registry import MODELS
from .base_mattor import BaseMattor

# For RGB !
group_norm_std = [0.229, 0.224, 0.225]
group_norm_mean = [0.485, 0.456, 0.406]


def np2torch(x):
    # to nchw
    return torch.from_numpy(x).permute(2, 0, 1)[None, :, :, :].float().cuda()


def groupnorm_normalise_image(img, format='nhwc'):
    """with format: batch, height, width, channel.

    Accept rgb in range 0,1
    """
    if (format == 'nhwc'):
        for i in range(3):
            img[...,
                i] = (img[..., i] - group_norm_mean[i]) / group_norm_std[i]
    else:
        for i in range(3):
            img[..., i, :, :] = (img[..., i, :, :] -
                                 group_norm_mean[i]) / group_norm_std[i]

    return img

def fba_fusion(alpha, img, F, B):
    F = ((alpha * img + (1 - alpha**2) * F - alpha * (1 - alpha) * B))
    B = ((1 - alpha) * img + (2 * alpha - alpha**2) * B - alpha *
         (1 - alpha) * F)

    F = torch.clamp(F, 0, 1)
    B = torch.clamp(B, 0, 1)
    la = 0.1
    alpha = (alpha * la + torch.sum((img - B) * (F - B), 1, keepdim=True)) / (
        torch.sum((F - B) * (F - B), 1, keepdim=True) + la)
    alpha = torch.clamp(alpha, 0, 1)
    return alpha, F, B



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
        loss_alpha (dict): Config of the alpha prediction loss. Default: None.
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
        super(FBA, self).__init__(backbone, None, train_cfg, test_cfg,
                                  pretrained)
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
                trimap,
                transformed_trimap,
                meta,
                fg=None,
                bg=None,
                alpha=None,
                test_mode=False,
                **kwargs):

        if not test_mode:
            return self.forward_train(merged, trimap, transformed_trimap,
                                      alpha, fg, bg, **kwargs)
        else:
            return self.forward_test(merged, trimap, transformed_trimap, meta,
                                     **kwargs)

    @auto_fp16(apply_to=('x', ))
    def _forward(self, x):
        output_fba = self.backbone(x)
        return output_fba

    def forward_dummy(self, inputs):
        return self._forward(inputs)

    def forward_train(self, merged, trimap, trimap_transformed, alpha, fg, bg,
                      ori_fg):
        """Forward function for training FBA model.

        Args:
            merged (Tensor): with shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            trimap (Tensor): with shape (N, C', H, W). Tensor of trimap. C'
                might be 1 or 3.
            trimap_transformed (Tensor): with shape (N, 2, H, W).
                Tensor of trimap.
            alpha (Tensor): with shape (N, 1, H, W). Tensor of alpha.
            fg (Tensor): with shape (N, 3, H, W). Tensor of fg.
                Fg extended to the whole image.
            bg (Tensor): with shape (N, 3, H, W). Tensor of bg.
            ori_fg (Tensor): with shape (N, 3, H, W). Tensor of ori_fg.

        Returns:
            dict: Contains the loss items and batch infomation.
        """
        merged_transformed = groupnorm_normalise_image(
            merged.clone(), format='nchw')
        input_tuple = tuple(
            (merged_transformed, trimap_transformed, trimap, merged))
        pred_alpha, pred_fg, pred_bg = self._forward(input_tuple)

        losses = dict()
        if self.loss_alpha is not None:
            losses['loss_alpha'] = self.loss_alpha(pred_alpha, alpha)
        if self.loss_alpha_compo is not None:
            losses['loss_alpha_compo'] = self.loss_alpha_compo(
                pred_alpha, ori_fg, bg, merged)
        if self.loss_alpha_lap is not None:
            losses['loss_alpha_lap'] = self.loss_alpha_lap(alpha, pred_alpha)

        if self.loss_fb is not None:
            losses['loss_fb'] = self.loss_fb(pred_fg, ori_fg) + self.loss_fb(
                pred_bg, bg)
        if self.loss_fb_compo is not None:
            losses['loss_fb_compo'] = self.loss_fb_compo(
                pred_fg, pred_bg, alpha, merged)
        if self.loss_exclusion is not None:
            losses['loss_exclusion'] = self.loss_exclusion(pred_bg, pred_fg)
        if self.loss_fb_lap is not None:
            losses['loss_fb_lap'] = self.loss_fb_lap(
                fg, pred_fg) + self.loss_fb_lap(bg, pred_bg)

        return {'losses': losses, 'num_samples': merged.size(0)}

    def forward_test(self,
                     merged,
                     trimap,
                     trimap_transformed,
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

        merged_transformed = groupnorm_normalise_image(
            merged.clone(), format='nchw')
        # for batch size 1
        input_tuple = tuple(
            (merged_transformed, trimap_transformed, trimap, merged))

        pred_alpha, pred_fg, pred_bg = self._forward(input_tuple)
        # FBA Fusion
        pred_alpha, pred_fg, pred_bg = fba_fusion(pred_alpha,merged, pred_fg, pred_bg)
        pred_fba = torch.cat((pred_alpha, pred_fg, pred_bg), 1)[0]
        ori_h, ori_w = meta[0]['merged_ori_shape'][:2]
        pred_fba = cv2.resize(pred_fba.cpu().numpy().transpose((1, 2, 0)),
                              (ori_w, ori_h), cv2.INTER_LANCZOS4)
        pred_fba = np2torch(pred_fba)[0]
        pred_a = pred_fba[0, ...]
        pred_b = pred_fba[1:4, ...]
        pred_f = pred_fba[4:7, ...]
        trimap_o = meta[0]['trimap_o']
        pred_a[trimap_o[..., 0] == 1] = 0
        pred_a[trimap_o[..., 1] == 1] = 1
        mask = pred_a[None, ...].repeat(3, 1, 1)
        pred_f[mask == 1] = np2torch(meta[0]['ori_merged'])[0][mask == 1]
        pred_b[mask == 0] = np2torch(meta[0]['ori_merged'])[0][mask == 0]
        pred_a = pred_a.detach().cpu().numpy().squeeze()
        # pred_b = pred_b.detach().cpu().numpy().squeeze()
        # pred_f = pred_f.detach().cpu().numpy().squeeze()
        eval_result = self.evaluate(pred_a, meta)

        if save_image:
            save_a = os.path.join(save_path, 'alpha')
            # save_b = os.path.join(save_path, 'bg')
            # save_f = os.path.join(save_path, 'fg')

            self.save_image(pred_a, meta, save_a, iteration)
            # self.save_image(pred_f, meta, save_f, iteration)
            # self.save_image(pred_b, meta, save_b, iteration)

        return {'pred_alpha': pred_a, 'eval_result': eval_result}
