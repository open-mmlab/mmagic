import logging
import os.path as osp
from abc import abstractmethod
from pathlib import Path

import mmcv
import numpy as np
from mmcv import ConfigDict
from mmcv.utils import print_log

from mmedit.core.evaluation import connectivity, gradient_error, mse, sad
from ..base import BaseModel
from ..builder import build_backbone, build_component
from ..registry import MODELS


@MODELS.register_module()
class BaseMattor(BaseModel):
    """Base class for matting model.

    A matting model must contain a backbone which produces `alpha`, a dense
    prediction with the same height and width of input image. In some cases,
    the model will has a refiner which refines the prediction of the backbone.

    The subclasses should overwrite the function ``forward_train`` and
    ``forward_test`` which define the output of the model and maybe the
    connection between the backbone and the refiner.

    Args:
        backbone (dict): Config of backbone.
        refiner (dict): Config of refiner.
        train_cfg (dict): Config of training. In ``train_cfg``,
            ``train_backbone`` should be specified. If the model has a refiner,
            ``train_refiner`` should be specified.
        test_cfg (dict): Config of testing. In ``test_cfg``, If the model has a
            refiner, ``train_refiner`` should be specified.
        pretrained (str): Path of pretrained model.
    """
    allowed_metrics = {
        'SAD': sad,
        'MSE': mse,
        'GRAD': gradient_error,
        'CONN': connectivity
    }

    def __init__(self,
                 backbone,
                 refiner=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(BaseMattor, self).__init__()

        self.train_cfg = train_cfg if train_cfg is not None else ConfigDict()
        self.test_cfg = test_cfg if test_cfg is not None else ConfigDict()

        self.backbone = build_backbone(backbone)
        # build refiner if it's not None.
        if refiner is None:
            self.train_cfg['train_refiner'] = False
            self.test_cfg['refine'] = False
        else:
            self.refiner = build_component(refiner)

        # if argument train_cfg is not None, validate if the config is proper.
        if train_cfg is not None:
            assert hasattr(self.train_cfg, 'train_refiner')
            assert hasattr(self.test_cfg, 'refine')
            if self.test_cfg.refine and not self.train_cfg.train_refiner:
                print_log(
                    'You are not training the refiner, but it is used for '
                    'model forwarding.', 'root', logging.WARNING)

            if not self.train_cfg.train_backbone:
                self.freeze_backbone()

        # validate if test config is proper
        if not hasattr(self.test_cfg, 'metrics'):
            raise KeyError('Missing key "metrics" in test_cfg')
        elif mmcv.is_list_of(self.test_cfg.metrics, str):
            for metric in self.test_cfg.metrics:
                if metric not in self.allowed_metrics:
                    raise KeyError(f'metric {metric} is not supported')
        elif self.test_cfg.metrics is not None:
            raise TypeError('metrics must be None or a list of str')

        self.init_weights(pretrained)

    @property
    def with_refiner(self):
        """Whether the matting model has a refiner.
        """
        return hasattr(self, 'refiner') and self.refiner is not None

    def freeze_backbone(self):
        """Freeze the backbone and only train the refiner.
        """
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the model network weights.

        Args:
            pretrained (str, optional): Path to the pretrained weight.
                Defaults to None.
        """
        if pretrained is not None:
            print_log(f'load model from: {pretrained}', logger='root')
        self.backbone.init_weights(pretrained)
        if self.with_refiner:
            self.refiner.init_weights()

    def restore_shape(self, pred_alpha, meta):
        """Restore the predicted alpha to the original shape.

        The shape of the predicted alpha may not be the same as the shape of
        original input image. This function restores the shape of the predicted
        alpha.

        Args:
            pred_alpha (np.ndarray): The predicted alpha.
            meta (list[dict]): Meta data about the current data batch.
                Currently only batch_size 1 is supported.

        Returns:
            np.ndarray: The reshaped predicted alpha.
        """
        ori_trimap = meta[0]['ori_trimap'].squeeze()
        ori_h, ori_w = meta[0]['merged_ori_shape'][:2]

        if 'interpolation' in meta[0]:
            # images have been resized for inference, resize back
            pred_alpha = mmcv.imresize(
                pred_alpha, (ori_w, ori_h),
                interpolation=meta[0]['interpolation'])
        elif 'pad' in meta[0]:
            # images have been padded for inference, remove the padding
            pred_alpha = pred_alpha[:ori_h, :ori_w]

        assert pred_alpha.shape == (ori_h, ori_w)

        # some methods do not have an activation layer after the last conv,
        # clip to make sure pred_alpha range from 0 to 1.
        pred_alpha = np.clip(pred_alpha, 0, 1)
        pred_alpha[ori_trimap == 0] = 0.
        pred_alpha[ori_trimap == 255] = 1.

        return pred_alpha

    def evaluate(self, pred_alpha, meta):
        """Evaluate predicted alpha matte.

        The evaluation metrics are determined by ``self.test_cfg.metrics``.

        Args:
            pred_alpha (np.ndarray): The predicted alpha matte of shape (H, W).
            meta (list[dict]): Meta data about the current data batch.
                Currently only batch_size 1 is supported. Required keys in the
                meta dict are ``ori_alpha`` and ``ori_trimap``.

        Returns:
            dict: The evaluation result.
        """
        if self.test_cfg.metrics is None:
            return None

        ori_alpha = meta[0]['ori_alpha'].squeeze()
        ori_trimap = meta[0]['ori_trimap'].squeeze()

        eval_result = dict()
        for metric in self.test_cfg.metrics:
            eval_result[metric] = self.allowed_metrics[metric](
                ori_alpha, ori_trimap,
                np.round(pred_alpha * 255).astype(np.uint8))
        return eval_result

    def save_image(self, pred_alpha, meta, save_path, iteration):
        """Save predicted alpha to file.

        Args:
            pred_alpha (np.ndarray): The predicted alpha matte of shape (H, W).
            meta (list[dict]): Meta data about the current data batch.
                Currently only batch_size 1 is supported. Required keys in the
                meta dict are ``merged_path``.
            save_path (str): The directory to save predicted alpha matte.
            iteration (int | None): If given as None, the saved alpha matte
                will have the same file name with ``merged_path`` in meta dict.
                If given as an int, the saved alpha matte would named with
                postfix ``_{iteration}.png``.
        """
        image_stem = Path(meta[0]['merged_path']).stem
        if iteration is None:
            save_path = osp.join(save_path, f'{image_stem}.png')
        else:
            save_path = osp.join(save_path,
                                 f'{image_stem}_{iteration + 1:06d}.png')
        mmcv.imwrite(pred_alpha * 255, save_path)

    @abstractmethod
    def forward_train(self, merged, trimap, alpha, **kwargs):
        """Defines the computation performed at every training call.

        Args:
            merged (Tensor): Image to predict alpha matte.
            trimap (Tensor): Trimap of the input image.
            alpha (Tensor): Ground-truth alpha matte.
        """
        pass

    @abstractmethod
    def forward_test(self, merged, trimap, meta, **kwargs):
        """Defines the computation performed at every test call.
        """
        pass

    def train_step(self, data_batch, optimizer):
        """Defines the computation and network update at every training call.

        Args:
            data_batch (torch.Tensor): Batch of data as input.
            optimizer (torch.optim.Optimizer): Optimizer of the model.

        Returns:
            dict: Output of ``train_step`` containing the logging variables \
                of the current data batch.
        """
        outputs = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(outputs.pop('losses'))

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        outputs.update({'log_vars': log_vars})
        return outputs

    def forward(self,
                merged,
                trimap,
                meta,
                alpha=None,
                test_mode=False,
                **kwargs):
        """Defines the computation performed at every call.

        Args:
            merged (Tensor): Image to predict alpha matte.
            trimap (Tensor): Trimap of the input image.
            meta (list[dict]): Meta data about the current data batch.
                Defaults to None.
            alpha (Tensor, optional): Ground-truth alpha matte.
                Defaults to None.
            test_mode (bool, optional): Whether in test mode. If ``True``, it
                will call ``forward_test`` of the model. Otherwise, it will
                call ``forward_train`` of the model. Defaults to False.

        Returns:
            dict: Return the output of ``self.forward_test`` if ``test_mode`` \
                are set to ``True``. Otherwise return the output of \
                ``self.forward_train``.
        """
        if not test_mode:
            return self.forward_train(merged, trimap, meta, alpha, **kwargs)
        else:
            return self.forward_test(merged, trimap, meta, **kwargs)
