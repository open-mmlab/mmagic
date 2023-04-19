# Copyright (c) OpenMMLab. All rights reserved.
import functools

import torch
from mmengine.model import BaseModule
from torch import nn

from mmagic.registry import MODELS


def get_norm_layer(norm_type='instance'):
    """Gets the normalization layer.

    Args:
        norm_type (str): Type of the normalization layer.

    Returns:
        norm_layer (BatchNorm2d or InstanceNorm2d or None):
            normalization layer. Default: instance
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' %
                                  norm_type)
    return norm_layer


@MODELS.register_module()
class WeightLayer(BaseModule):
    """Weight layer of the fusion_net. A small neural network with three
    convolutional layers to predict full-image weight map and perinstance
    weight map.

    Args:
        input_ch (int): Number of channels in the input image.
        inner_ch (int): Number of channels produced by the convolution.
            Default: True
    """

    def __init__(self, input_ch, inner_ch=16):
        super().__init__()
        self.simple_instance_conv = nn.Sequential(
            nn.Conv2d(input_ch, inner_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(inner_ch, inner_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(inner_ch, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )

        self.simple_bg_conv = nn.Sequential(
            nn.Conv2d(input_ch, inner_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(inner_ch, inner_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(inner_ch, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )

        self.normalize = nn.Softmax(1)

    def resize_and_pad(self, feauture_maps, info_array):
        """Resize the instance feature as well as the weight map to match the
        size of full-image and do zero padding on both of them.

        Args:
            feauture_maps (tensor): Feature map
            info_array (tensor): The bounding box

        Returns:
            feauture_maps (tensor): Feature maps after resize and padding
        """
        feauture_maps = torch.nn.functional.interpolate(
            feauture_maps,
            size=(info_array[5], info_array[4]),
            mode='bilinear')
        feauture_maps = torch.nn.functional.pad(feauture_maps,
                                                (info_array[0], info_array[1],
                                                 info_array[2], info_array[3]),
                                                'constant', 0)
        return feauture_maps

    def forward(self, instance_feature, bg_feature, box_info):
        """Forward function.

        Args:
            instance_feature (tensor): Instance feature obtained from the
                colorization_net
            bg_feature (tensor):  full-image feature
            box_info (tensor): The bounding box corresponding to the instance

        Returns:
            out (tensor): Fused feature
        """
        mask_list = []
        featur_map_list = []
        mask_sum_for_pred = torch.zeros_like(bg_feature)[:1, :1]
        for i in range(instance_feature.shape[0]):
            tmp_crop = torch.unsqueeze(instance_feature[i], 0)
            conv_tmp_crop = self.simple_instance_conv(tmp_crop)
            pred_mask = self.resize_and_pad(conv_tmp_crop, box_info[i])

            tmp_crop = self.resize_and_pad(tmp_crop, box_info[i])

            mask = torch.zeros_like(bg_feature)[:1, :1]
            mask[0, 0, box_info[i][2]:box_info[i][2] + box_info[i][5],
                 box_info[i][0]:box_info[i][0] + box_info[i][4]] = 1.0
            device = mask.device
            mask = mask.type(torch.FloatTensor).to(device)

            mask_sum_for_pred = torch.clamp(mask_sum_for_pred + mask, 0.0, 1.0)

            mask_list.append(pred_mask)
            featur_map_list.append(tmp_crop)

        pred_bg_mask = self.simple_bg_conv(bg_feature)
        mask_list.append(pred_bg_mask + (1 - mask_sum_for_pred) * 100000.0)
        mask_list = self.normalize(torch.cat(mask_list, 1))

        mask_list_maskout = mask_list.clone()

        featur_map_list.append(bg_feature)
        featur_map_list = torch.cat(featur_map_list, 0)
        mask_list_maskout = mask_list_maskout.permute(1, 0, 2, 3).contiguous()
        out = featur_map_list * mask_list_maskout
        out = torch.sum(out, 0, keepdim=True)
        return out
