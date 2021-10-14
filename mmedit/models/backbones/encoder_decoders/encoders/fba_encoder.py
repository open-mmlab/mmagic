# Copyright (c) OpenMMLab. All rights reserved.
from mmedit.models.registry import COMPONENTS
from .resnet import ResNet


@COMPONENTS.register_module()
class FBAResnetDilated(ResNet):
    """ResNet-based encoder for FBA image matting."""

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (N, C, H, W).

        Returns:
            Tensor: Output tensor.
        """
        # x: (merged_t, trimap_t, two_channel_trimap,merged)
        # t refers to transformed.
        two_channel_trimap = x[:, 9:11]
        merged = x[:, 11:14]
        x = x[:, 0:11, ...]
        conv_out = [x]
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.activate(x)
        conv_out.append(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        conv_out.append(x)
        x = self.layer2(x)
        conv_out.append(x)
        x = self.layer3(x)
        conv_out.append(x)
        x = self.layer4(x)
        conv_out.append(x)
        return {
            'conv_out': conv_out,
            'merged': merged,
            'two_channel_trimap': two_channel_trimap
        }
