import logging
from abc import abstractmethod

import mmcv
from mmcv import ConfigDict
from mmcv.utils import print_log
from mmedit.core.evaluation import mse, sad

from ..base import BaseModel
from ..builder import build_backbone, build_component
from ..registry import MODELS


@MODELS.register_module
class BaseMattor(BaseModel):
    """Base class for matting model.

    A matting model must contain a backbone which produces `alpha`, a dense
    prediction with the same height and width of input image. In some cases,
    the model will has a refiner which refines the prediction of backbone.

    The subclasses should overwrite the function `forward_train` and
    `forward_test` which define the output of the model and maybe the
    connection between backbone and refiner.

    Args:
        backbone (dict): Config of backbone.
        refiner (dict): Config of refiner.
        train_cfg (dict): Config of training. In 'train_cfg', 'train_backbone'
            should be specified. If the model has a refiner, 'train_refiner'
            should be specified.
        test_cfg (dict): Config of testing. In 'test_cfg', If the model has a
            refiner, 'train_refiner' should be specified.
        pretrained (str): path of pretrained model.
    """
    allowed_metrics = {'SAD': sad, 'MSE': mse}

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
        return hasattr(self, 'refiner') and self.refiner is not None

    def freeze_backbone(self):
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            print_log(f'load model from: {pretrained}', logger='root')
        self.backbone.init_weights(pretrained)
        if self.with_refiner:
            self.refiner.init_weights()

    def restore_shape(self, pred_alpha, img_meta):
        """Restore predicted alpha to origin shape.

        The shape of the predicted alpha may not be the same as the shape of
        origin input image. This function restore the shape of predicted alpha.

        Args:
            pred_alpha (np.ndarray): The predicted alpha.
            img_meta (list[dict]): Meta data about the current data batch.
                Currently only batch_size 1 is supported.

        Returns:
            ndarray: The reshaped predicted alpha.
        """
        ori_trimap = img_meta[0]['ori_trimap']
        ori_h, ori_w = img_meta[0]['ori_shape']
        pred_alpha = mmcv.imresize(pred_alpha, (ori_w, ori_h))
        pred_alpha[ori_trimap == 0] = 0.
        pred_alpha[ori_trimap == 255] = 1.

        return pred_alpha

    def evaluate(self, pred_alpha, img_meta):
        if self.test_cfg.metrics is None:
            return None

        ori_alpha = img_meta[0]['ori_alpha']
        ori_trimap = img_meta[0]['ori_trimap']

        eval_result = dict()
        for metric in self.test_cfg.metrics:
            eval_result[metric] = self.allowed_metrics[metric](ori_alpha,
                                                               ori_trimap,
                                                               pred_alpha)
        return eval_result

    @abstractmethod
    def forward_train(self, merged, trimap, alpha, **kwargs):
        pass

    @abstractmethod
    def forward_test(self, merged, trimap, img_meta, **kwargs):
        pass

    def train_step(self, data_batch, optimizer):
        raise NotImplementedError('train_step should not be used in mattors.')

    def forward(self,
                merged,
                trimap,
                alpha=None,
                img_meta=None,
                test_mode=False,
                **kwargs):
        if not test_mode:
            return self.forward_train(merged, trimap, alpha, **kwargs)
        else:
            return self.forward_test(merged, trimap, img_meta, **kwargs)
