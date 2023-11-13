# Copyright (c) OpenMMLab. All rights reserved.
import mmengine
import torch
from torch import nn

from mmagic.registry import MODELS
from .model_irse import Backbone


@MODELS.register_module('ArcFace')
class IDLossModel(nn.Module):
    """Face id loss model.

    Args:
        ir_se50_weights (str, optional): Url of ir-se50 weights.
            Defaults to None.
    """
    # ir se50 weight download link
    _ir_se50_url = 'https://download.openxlab.org.cn/models/rangoliu/Arcface-IR-SE50/weight/Arcface-IR-SE50'  # noqa

    def __init__(self, ir_se50_weights=None):
        super(IDLossModel, self).__init__()
        mmengine.print_log('Loading ResNet ArcFace', 'current')
        self.facenet = Backbone(
            input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        if ir_se50_weights is None:
            ir_se50_weights = self._ir_se50_url
        self.facenet.load_state_dict(
            torch.hub.load_state_dict_from_url(
                ir_se50_weights, map_location='cpu'))
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet = self.facenet.eval()

    def extract_feats(self, x):
        """Extracting face features.

        Args:
            x (torch.Tensor): Image tensor of faces.

        Returns:
            torch.Tensor: Face features.
        """
        if x.shape[2] != 256:
            x = self.pool(x)
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, pred=None, gt=None):
        """Calculate face loss.

        Args:
            pred (torch.Tensor, optional): Predictions of face images.
                Defaults to None.
            gt (torch.Tensor, optional): Ground truth of face images.
                Defaults to None.

        Returns:
            Tuple(float, float): A tuple contain face similarity loss and
                improvement.
        """
        n_samples = gt.shape[0]
        y_feats = self.extract_feats(
            gt)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(pred)
        y_feats = y_feats.detach()
        loss = 0
        sim_improvement = 0
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            loss += 1 - diff_target
            count += 1

        return loss / count, sim_improvement / count
