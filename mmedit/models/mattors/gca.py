import torch

from ..builder import build_loss
from ..registry import MODELS
from . import BaseMattor


def get_unknown_tensor(trimap):
    """Get 1-channel unknown area tensor from the 3 or 1-channel trimap tensor.

    Args:
        trimap (Tensor): with shape (N, 3, H, W) or (N, 1, H, W).

    Returns:
        Tensor: Unknown area mask of shape (N, 1, H, W).
    """
    if trimap.shape[1] == 3:
        # The three channels correspond to (bg mask, unknown mask, fg mask)
        # respectively.
        weight = trimap[:, 1:2, :, :].float()
    else:
        # 0 for bg, 1 for unknown, 2 for fg
        weight = trimap.eq(1).float()
    return weight


@MODELS.register_module
class GCA(BaseMattor):
    """Guided Contextual Attention image matting model.

    https://arxiv.org/abs/2001.04069

    Args:
        backbone (dict): Config of backbone.
        train_cfg (dict): Config of training. In 'train_cfg', 'train_backbone'
            should be specified. If the model has a refiner, 'train_refiner'
            should be specified.
        test_cfg (dict): Config of testing. In 'test_cfg', If the model has a
            refiner, 'train_refiner' should be specified.
        pretrained (str): Path of pretrained model.
        loss_alpha (dict): Config of the alpha prediction loss. Default: None.
    """

    def __init__(self,
                 backbone,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 loss_alpha=None):
        super(GCA, self).__init__(backbone, None, train_cfg, test_cfg,
                                  pretrained)
        self.loss_alpha = build_loss(loss_alpha)

    def _forward(self, x):
        raw_alpha = self.backbone(x)
        pred_alpha = (raw_alpha.tanh() + 1.0) / 2.0
        return pred_alpha

    def forward_dummy(self, inputs):
        return self._forward(inputs)

    def forward_train(self, merged, trimap, alpha):
        """Forward function for training GCA model.

        Args:
            merged (Tensor): with shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            trimap (Tensor): with shape (N, 1, H, W). Tensor of trimap.
            alpha (Tensor): with shape (N, 1, H, W). Tensor of alpha.

        Returns:
            dict: Contains the loss items and batch infomation.
        """
        pred_alpha = self._forward(torch.cat((merged, trimap), 1))

        weight = get_unknown_tensor(trimap)
        losses = {'loss': self.loss_alpha(pred_alpha, alpha, weight)}
        return {'losses': losses, 'num_samples': merged.size(0)}

    def forward_test(self,
                     merged,
                     trimap,
                     meta,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        pred_alpha = self._forward(torch.cat((merged, trimap), 1))

        pred_alpha = pred_alpha.cpu().numpy().squeeze()
        pred_alpha = self.restore_shape(pred_alpha, meta)
        eval_result = self.evaluate(pred_alpha, meta)

        if save_image:
            self.save_image(pred_alpha, meta, save_path, iteration)

        return {'eval_result': eval_result}
