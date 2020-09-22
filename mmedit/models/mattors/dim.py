import torch
from mmcv.runner import auto_fp16

from ..builder import build_loss
from ..registry import MODELS
from .base_mattor import BaseMattor
from .utils import get_unknown_tensor


@MODELS.register_module()
class DIM(BaseMattor):
    """Deep Image Matting model.

    https://arxiv.org/abs/1703.03872

    .. note::

        For ``(self.train_cfg.train_backbone, self.train_cfg.train_refiner)``:

            * ``(True, False)`` corresponds to the encoder-decoder stage in \
                the paper.
            * ``(False, True)`` corresponds to the refinement stage in the \
                paper.
            * ``(True, True)`` corresponds to the fine-tune stage in the paper.

    Args:
        backbone (dict): Config of backbone.
        refiner (dict): Config of refiner.
        train_cfg (dict): Config of training. In ``train_cfg``,
            ``train_backbone`` should be specified. If the model has a refiner,
            ``train_refiner`` should be specified.
        test_cfg (dict): Config of testing. In ``test_cfg``, If the model has a
            refiner, ``train_refiner`` should be specified.
        pretrained (str): Path of pretrained model.
        loss_alpha (dict): Config of the alpha prediction loss. Default: None.
        loss_comp (dict): Config of the composition loss. Default: None.
        loss_refine (dict): Config of the loss of the refiner. Default: None.
    """

    def __init__(self,
                 backbone,
                 refiner=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 loss_alpha=None,
                 loss_comp=None,
                 loss_refine=None):
        super(DIM, self).__init__(backbone, refiner, train_cfg, test_cfg,
                                  pretrained)

        if all(v is None for v in (loss_alpha, loss_comp, loss_refine)):
            raise ValueError('Please specify one loss for DIM.')

        if loss_alpha is not None:
            self.loss_alpha = build_loss(loss_alpha)
        if loss_comp is not None:
            self.loss_comp = build_loss(loss_comp)
        if loss_refine is not None:
            self.loss_refine = build_loss(loss_refine)

        # support fp16
        self.fp16_enabled = False

    @auto_fp16()
    def _forward(self, x, refine):
        raw_alpha = self.backbone(x)
        pred_alpha = raw_alpha.sigmoid()

        if refine:
            refine_input = torch.cat((x[:, :3, :, :], pred_alpha), 1)
            pred_refine = self.refiner(refine_input, raw_alpha)
        else:
            # As ONNX does not support NoneType for output,
            # we choose to use zero tensor to represent None
            pred_refine = torch.zeros([])
        return pred_alpha, pred_refine

    def forward_dummy(self, inputs):
        return self._forward(inputs, self.with_refiner)

    def forward_train(self, merged, trimap, meta, alpha, ori_merged, fg, bg):
        """Defines the computation performed at every training call.

        Args:
            merged (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            trimap (Tensor): of shape (N, 1, H, W). Tensor of trimap read by
                opencv.
            meta (list[dict]): Meta data about the current data batch.
            alpha (Tensor): of shape (N, 1, H, W). Tensor of alpha read by
                opencv.
            ori_merged (Tensor): of shape (N, C, H, W). Tensor of origin merged
                image read by opencv (not normalized).
            fg (Tensor): of shape (N, C, H, W). Tensor of fg read by opencv.
            bg (Tensor): of shape (N, C, H, W). Tensor of bg read by opencv.

        Returns:
            dict: Contains the loss items and batch infomation.
        """
        pred_alpha, pred_refine = self._forward(
            torch.cat((merged, trimap), 1), self.train_cfg.train_refiner)

        weight = get_unknown_tensor(trimap, meta)
        losses = dict()
        if self.train_cfg.train_backbone:
            if self.loss_alpha is not None:
                losses['loss_alpha'] = self.loss_alpha(pred_alpha, alpha,
                                                       weight)
            if self.loss_comp is not None:
                losses['loss_comp'] = self.loss_comp(pred_alpha, fg, bg,
                                                     ori_merged, weight)
        if self.train_cfg.train_refiner:
            losses['loss_refine'] = self.loss_refine(pred_refine, alpha,
                                                     weight)
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
        pred_alpha, pred_refine = self._forward(
            torch.cat((merged, trimap), 1), self.test_cfg.refine)
        if self.test_cfg.refine:
            pred_alpha = pred_refine

        pred_alpha = pred_alpha.detach().cpu().numpy().squeeze()
        pred_alpha = self.restore_shape(pred_alpha, meta)
        eval_result = self.evaluate(pred_alpha, meta)

        if save_image:
            self.save_image(pred_alpha, meta, save_path, iteration)

        return {'pred_alpha': pred_alpha, 'eval_result': eval_result}
