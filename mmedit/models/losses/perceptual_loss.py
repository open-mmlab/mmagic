# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torchvision.models.vgg as vgg
from mmcv.runner import load_checkpoint
from torch.nn import functional as F

from mmedit.utils import get_root_logger
from ..registry import LOSSES


class PerceptualVGG(nn.Module):
    """VGG network used in calculating perceptual loss.

    In this implementation, we allow users to choose whether use normalization
    in the input feature and the type of vgg network. Note that the pretrained
    path must fit the vgg type.

    Args:
        layer_name_list (list[str]): According to the name in this list,
            forward function will return the corresponding features. This
            list contains the name each layer in `vgg.feature`. An example
            of this list is ['4', '10'].
        vgg_type (str): Set the type of vgg network. Default: 'vgg19'.
        use_input_norm (bool): If True, normalize the input image.
            Importantly, the input feature must in the range [0, 1].
            Default: True.
        pretrained (str): Path for pretrained weights. Default:
            'torchvision://vgg19'
    """

    def __init__(self,
                 layer_name_list,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 pretrained='torchvision://vgg19'):
        super().__init__()
        if pretrained.startswith('torchvision://'):
            assert vgg_type in pretrained
        self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm

        # get vgg model and load pretrained vgg weight
        # remove _vgg from attributes to avoid `find_unused_parameters` bug
        _vgg = getattr(vgg, vgg_type)()
        self.init_weights(_vgg, pretrained)
        num_layers = max(map(int, layer_name_list)) + 1
        assert len(_vgg.features) >= num_layers
        # only borrow layers that will be used from _vgg to avoid unused params
        self.vgg_layers = _vgg.features[:num_layers]

        if self.use_input_norm:
            # the mean is for image with range [0, 1]
            self.register_buffer(
                'mean',
                torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            # the std is for image with range [-1, 1]
            self.register_buffer(
                'std',
                torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        for v in self.vgg_layers.parameters():
            v.requires_grad = False

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = {}

        for name, module in self.vgg_layers.named_children():
            x = module(x)
            if name in self.layer_name_list:
                output[name] = x.clone()
        return output

    def init_weights(self, model, pretrained):
        """Init weights.

        Args:
            model (nn.Module): Models to be inited.
            pretrained (str): Path for pretrained weights.
        """
        logger = get_root_logger()
        load_checkpoint(model, pretrained, logger=logger)


@LOSSES.register_module()
class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layers_weights (dict): The weight for each layer of vgg feature for
            perceptual loss. Here is an example: {'4': 1., '9': 1., '18': 1.},
            which means the 5th, 10th and 18th feature layer will be
            extracted with weight 1.0 in calculating losses.
        layers_weights_style (dict): The weight for each layer of vgg feature
            for style loss. If set to 'None', the weights are set equal to
            the weights for perceptual loss. Default: None.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 1.0.
        norm_img (bool): If True, the image will be normed to [0, 1]. Note that
            this is different from the `use_input_norm` which norm the input in
            in forward function of vgg according to the statistics of dataset.
            Importantly, the input image must be in range [-1, 1].
        pretrained (str): Path for pretrained weights. Default:
            'torchvision://vgg19'.
        criterion (str): Criterion type. Options are 'l1' and 'mse'.
            Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 layer_weights_style=None,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 perceptual_weight=1.0,
                 style_weight=1.0,
                 norm_img=True,
                 pretrained='torchvision://vgg19',
                 criterion='l1'):
        super().__init__()
        self.norm_img = norm_img
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.layer_weights_style = layer_weights_style

        self.vgg = PerceptualVGG(
            layer_name_list=list(self.layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            pretrained=pretrained)

        if self.layer_weights_style is not None and \
                self.layer_weights_style != self.layer_weights:
            self.vgg_style = PerceptualVGG(
                layer_name_list=list(self.layer_weights_style.keys()),
                vgg_type=vgg_type,
                use_input_norm=use_input_norm,
                pretrained=pretrained)
        else:
            self.layer_weights_style = self.layer_weights
            self.vgg_style = None

        criterion = criterion.lower()
        if criterion == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif criterion == 'mse':
            self.criterion = torch.nn.MSELoss()
        else:
            raise NotImplementedError(
                f'{criterion} criterion has not been supported in'
                ' this version.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        if self.norm_img:
            x = (x + 1.) * 0.5
            gt = (gt + 1.) * 0.5
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                percep_loss += self.criterion(
                    x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            if self.vgg_style is not None:
                x_features = self.vgg_style(x)
                gt_features = self.vgg_style(gt.detach())

            style_loss = 0
            for k in x_features.keys():
                style_loss += self.criterion(
                    self._gram_mat(x_features[k]),
                    self._gram_mat(
                        gt_features[k])) * self.layer_weights_style[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        (n, c, h, w) = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram


@LOSSES.register_module()
class TransferalPerceptualLoss(nn.Module):
    """Transferal perceptual loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
        use_attention (bool): If True, use soft-attention tensor. Default: True
        criterion (str): Criterion type. Options are 'l1' and 'mse'.
            Default: 'l1'.
    """

    def __init__(self, loss_weight=1.0, use_attention=True, criterion='mse'):
        super().__init__()
        self.use_attention = use_attention
        self.loss_weight = loss_weight
        criterion = criterion.lower()
        if criterion == 'l1':
            self.loss_function = torch.nn.L1Loss()
        elif criterion == 'mse':
            self.loss_function = torch.nn.MSELoss()
        else:
            raise ValueError(
                f"criterion should be 'l1' or 'mse', but got {criterion}")

    def forward(self, maps, soft_attention, textures):
        """Forward function.

        Args:
            maps (Tuple[Tensor]): Input tensors.
            soft_attention (Tensor): Soft-attention tensor.
            textures (Tuple[Tensor]): Ground-truth tensors.

        Returns:
            Tensor: Forward results.
        """

        if self.use_attention:
            h, w = soft_attention.shape[-2:]
            softs = [torch.sigmoid(soft_attention)]
            for i in range(1, len(maps)):
                softs.append(
                    F.interpolate(
                        soft_attention,
                        size=(h * pow(2, i), w * pow(2, i)),
                        mode='bicubic',
                        align_corners=False))
        else:
            softs = [1., 1., 1.]

        loss_texture = 0
        for map, soft, texture in zip(maps, softs, textures):
            loss_texture += self.loss_function(map * soft, texture * soft)

        return loss_texture * self.loss_weight
