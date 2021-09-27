import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
from torchvision import models

from mmedit.models.registry import BACKBONES


def get_mean_std(feat, eps=1e-5):
    """get mean and std of feature

    Args:
        feat (Tensor): Input tensor with shape (n, c, h, w).
        eps (float): A small value added to the variance to avoid
            divide-by-zero.

    Returns:
        feat_mean (Tensor): Mean of feature (n, c, 1, 1).
        feat_std (Tensor): Std of feature (n, c, 1, 1).
    """

    size = feat.size()
    assert len(size) == 4
    n, c = size[:2]
    feat_var = feat.view(n, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(n, c, 1, 1)
    feat_mean = feat.view(n, c, -1).mean(dim=2).view(n, c, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    """Normalize one feature with reference to another feature.

    Args:
        content_feat (Tensor): ref feature with shape (n, c, h, w).
        style_feat (float): degradate feature with shape (n, c, h, w).

    Returns:
        Tensor: Normalized feature (n, c, h, w).
    """

    size = content_feat.size()
    style_mean, style_std = get_mean_std(style_feat)

    content_mean, content_std = get_mean_std(content_feat)
    normalized_feat = (content_feat -
                       content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class BasicBlock(nn.Sequential):
    """Basic Block of DFD

    Args:
        in_channels (int): Number of channels in the input feature.
        kernel_size (int | Tuple[int]): Size of the convolving kernel.
            Default: 3
        stride (int | Tuple[int]): Stride of the convolution. Default: 1
        dilation (int | Tuple[int]): Spacing between kernel elements.
            Default: 1
        bias (bool): If 'True', adds a learnable bias to the output.
            Default: 'True'
        out_channels (int | None): Number of channels in the output feature.
            If None, out_channels = in_channels.
            Default: None.
        mid_channels (int | None): Channel number of intermediate features.
            If None, mid_channels = in_channels.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 bias=True,
                 out_channels=None,
                 mid_channels=None):
        if out_channels is None:
            out_channels = in_channels
        if mid_channels is None:
            mid_channels = in_channels
        padding = ((kernel_size - 1) // 2) * dilation
        modules = [
            spectral_norm(
                nn.Conv2d(
                    in_channels,
                    mid_channels,
                    kernel_size,
                    stride,
                    padding,
                    dilation=dilation,
                    bias=bias)),
            nn.LeakyReLU(0.2),
            spectral_norm(
                nn.Conv2d(
                    mid_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    dilation=dilation,
                    bias=bias))
        ]
        super().__init__(*modules)


class MSDilateBlock(nn.Module):
    """Multi Scale Dilate Block of DFD

    Args:
        in_channels (Tensor): Number of channels in the input feature.
        kernel_size (int | Tuple[int]): Size of the convolving kernel.
            Default: 3
        dilations (Tuple[int]): Dilation of each layer. Default: [1,1,1,1]
        bias (bool): If 'True', adds a learnable bias to the output.
            Default: 'True'
    """

    def __init__(self,
                 in_channels,
                 kernel_size=3,
                 dilations=[1, 1, 1, 1],
                 bias=True):
        super().__init__()
        self.conv1 = BasicBlock(
            in_channels, kernel_size, dilation=dilations[0], bias=bias)
        self.conv2 = BasicBlock(
            in_channels, kernel_size, dilation=dilations[1], bias=bias)
        self.conv3 = BasicBlock(
            in_channels, kernel_size, dilation=dilations[2], bias=bias)
        self.conv4 = BasicBlock(
            in_channels, kernel_size, dilation=dilations[3], bias=bias)
        self.conv_last = spectral_norm(
            nn.Conv2d(
                in_channels=in_channels * 4,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                bias=bias))

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor.

        returns:
            Tensor: Forward results.
        """

        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        cat = torch.cat([conv1, conv2, conv3, conv4], 1)
        out = self.conv_last(cat) + x
        return out


class ResBlock(nn.Module):
    """Res Block with LeakyReLU.

    Args:
        mid_channels (int): Channel number of intermediate features.
    """

    def __init__(self, mid_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1))

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor.

        returns:
            Tensor: Forward results.
        """
        out = x + self.layers(x)
        return out


class BlurFunctionBackward(Function):

    @staticmethod
    def forward(ctx, grad_output, kernel, kernel_flip):
        """Forward Function.

        Args:
            ctx (BlurFunctionBackward): Store tensors that can be then
                retrieved during the backward pass.
            grad_output (Tensor): Tensor of grad output.
            kernel (int | Tuple[int]): Size of the convolving kernel.
            kernel_flip (int | Tuple[int]) : Size of flipped kernel.

        returns:
            grad_input (Tensor): Tensor of grad input.
        """
        ctx.save_for_backward(kernel, kernel_flip)

        grad_input = F.conv2d(
            grad_output, kernel_flip, padding=1, groups=grad_output.shape[1])
        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_output):
        """Backward Function.

        Args:
            ctx (BlurFunctionBackward): Store tensors that can be then
                retrieved during the backward pass.
            gradgrad_output (Tensor): Grad of grad output.

        returns:
            grad_input (Tensor): Tensor of grad input.
        """
        kernel, _ = ctx.saved_tensors

        grad_input = F.conv2d(
            gradgrad_output,
            kernel,
            padding=1,
            groups=gradgrad_output.shape[1])
        return grad_input, None, None


class BlurFunction(Function):

    @staticmethod
    def forward(ctx, input, kernel, kernel_flip):
        """Backward Function.

        Args:
            ctx (BlurFunctionBackward): Store tensors that can be then
                retrieved during the backward pass.
            input (Tensor): Input tensor.
            kernel (int | Tuple[int]): Size of the convolving kernel.
            kernel_flip (int | Tuple[int]) : Size of flipped kernel.

        returns:
            output (Tensor): Output tensor.
        """

        ctx.save_for_backward(kernel, kernel_flip)
        output = F.conv2d(input, kernel, padding=1, groups=input.shape[1])

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward Function.

        Args:
            ctx (BlurFunctionBackward): Store tensors that can be then
                retrieved during the backward pass.
            grad_output (Tensor): Tensor of grad output.

        returns:
            grad_input (Tensor): Tensor of grad input.
        """

        kernel, kernel_flip = ctx.saved_tensors

        grad_input = BlurFunctionBackward.apply(grad_output, kernel,
                                                kernel_flip)

        return grad_input, None, None


blur = BlurFunction.apply


class Blur(nn.Module):
    """Blur Layer.

    Args:
        channels (int): Channels of layer, in_channels=out_channels
    """

    def __init__(self, channels):
        super().__init__()

        weight = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]],
                              dtype=torch.float32)
        weight = weight.view(1, 1, 3, 3)
        weight = weight / weight.sum()
        weight_flip = torch.flip(weight, [2, 3])

        self.register_buffer('weight', weight.repeat(channels, 1, 1, 1))
        self.register_buffer('weight_flip',
                             weight_flip.repeat(channels, 1, 1, 1))
        self.blur = BlurFunction.apply

    def forward(self, input):
        """Forward Function.

        Args:
            input (Tensor): Input tensor.

        returns:
            Tensor: Forward results.
        """
        return blur(input, self.weight, self.weight_flip)


class StyledUpBlock(nn.Module):
    """Styled Upscale Block.

    Args:
        in_channels (int): Number of channels in the input features.
        mid_channels (int): Number of channels in the intermediate features.
        kernel_size (int | Tuple[int]): Size of the convolving kernel.
        padding (int | Tuple[int]): Zero-padding added to both sides of the
            input. Default: 0
        upsample (bool): Upsample or not. Default: False.
    """

    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size=3,
                 padding=1,
                 upsample=False):
        super().__init__()

        if upsample:
            self.conv_first = nn.Sequential(
                nn.Upsample(
                    scale_factor=2, mode='bilinear', align_corners=False),
                Blur(out_channel),
                spectral_norm(
                    nn.Conv2d(
                        in_channel, out_channel,
                        kernel_size, padding=padding)), nn.LeakyReLU(0.2),
                nn.LeakyReLU(0.2))
        else:
            self.conv_first = nn.Sequential(
                Blur(in_channel),
                spectral_norm(
                    nn.Conv2d(
                        in_channel, out_channel,
                        kernel_size, padding=padding)), nn.LeakyReLU(0.2),
                nn.LeakyReLU(0.2))
        self.conv_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            spectral_norm(
                nn.Conv2d(
                    out_channel, out_channel, kernel_size, padding=padding)),
            nn.LeakyReLU(0.2))

        self.scale_model = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channel, out_channel, 3, 1, 1)),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(out_channel, out_channel, 3, 1, 1)))
        self.shift_model = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channel, out_channel, 3, 1, 1)),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(out_channel, out_channel, 3, 1, 1)),
            nn.Sigmoid())

    def forward(self, input, style):
        """Forward Function.

        Args:
            input (Tensor): Input tensor.
            style (Tensor): Style tensor.

        Returns:
            Tensor: Forward results.
        """

        out = self.conv_first(input)
        shift = self.shift_model(style)
        scale = self.scale_model(style)
        out = out * scale + shift
        out_up = self.conv_up(out)

        return out_up


class VGGFeat(torch.nn.Module):
    """Encoder based on VGG.
    """

    def __init__(self, load_path):
        super().__init__()
        self.model = models.vgg19(pretrained=False)
        if load_path:
            self.model.load_state_dict(torch.load(load_path))
        self.build_vgg_layers()

        self.register_parameter(
            'rgb_mean',
            nn.Parameter(torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)))
        self.register_parameter(
            'rgb_std',
            nn.Parameter(torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)))

        for param in self.model.parameters():
            param.requires_grad = False

    def build_vgg_layers(self):
        """Build VGG Layers.
        """
        vgg_pretrained_features = self.model.features
        self.features = []
        feature_layers = [0, 8, 17, 26, 35]
        for i in range(len(feature_layers) - 1):
            module_layers = torch.nn.Sequential()
            for j in range(feature_layers[i], feature_layers[i + 1]):
                module_layers.add_module(str(j), vgg_pretrained_features[j])
            self.features.append(module_layers)
        self.features = torch.nn.ModuleList(self.features)

    def preprocess(self, x):
        """Modify the mean and variance.

        Args:
            x (Tensor): Input tensor.

        Results:
            Tensor: Processed tensor.
        """
        # print('save')
        # torch.save(self.state_dict(), 'work_dirs/vgg_load.pth')
        x = (x + 1) / 2
        x = (x - self.rgb_mean) / self.rgb_std
        if x.shape[3] < 224:
            x = torch.nn.functional.interpolate(
                x, size=(224, 224), mode='bilinear', align_corners=False)
        return x

    def forward(self, x):
        """Forward Function.

        Args:
            x (Tensor): Input tensor.

        Results:
            Tensor: Forward results.
        """

        x = self.preprocess(x)
        features = []
        for m in self.features:
            x = m(x)
            features.append(x)
        return features


@BACKBONES.register_module()
class DFDNet(nn.Module):
    """DIC network structure for face super-resolution.

    Paper: Blind Face Restoration via Deep Multi-scale Component Dictionaries.

    Args:
        dictionary (dict): Facial features dictionary.
        mid_channels (int):  Number of channels in the intermediate features.
            Default: 64
    """

    def __init__(self, mid_channels=64, vgg_load_path=None):
        super().__init__()

        self.part_sizes = np.array([80, 80, 50, 110])  # size for 512
        self.feature_sizes = [256, 128, 64, 32]

        self.left_eye_256 = BasicBlock(128)
        self.left_eye_128 = BasicBlock(256)
        self.left_eye_64 = BasicBlock(512)
        self.left_eye_32 = BasicBlock(512)

        self.right_eye_256 = BasicBlock(128)
        self.right_eye_128 = BasicBlock(256)
        self.right_eye_64 = BasicBlock(512)
        self.right_eye_32 = BasicBlock(512)

        self.nose_256 = BasicBlock(128)
        self.nose_128 = BasicBlock(256)
        self.nose_64 = BasicBlock(512)
        self.nose_32 = BasicBlock(512)

        self.mouth_256 = BasicBlock(128)
        self.mouth_128 = BasicBlock(256)
        self.mouth_64 = BasicBlock(512)
        self.mouth_32 = BasicBlock(512)

        self.vgg_extract = VGGFeat(vgg_load_path)
        vgg_mid_channels = 64  # locked by VGG
        self.ms_dilate = MSDilateBlock(
            vgg_mid_channels * 8, dilations=[4, 3, 2, 1])

        self.up0 = StyledUpBlock(vgg_mid_channels * 8, vgg_mid_channels * 8)
        self.up1 = StyledUpBlock(vgg_mid_channels * 8, vgg_mid_channels * 4)
        self.up2 = StyledUpBlock(vgg_mid_channels * 4, vgg_mid_channels * 2)
        self.up3 = StyledUpBlock(vgg_mid_channels * 2, vgg_mid_channels)
        self.up4 = nn.Sequential(
            spectral_norm(
                nn.Conv2d(vgg_mid_channels, vgg_mid_channels, 3, 1, 1)),
            nn.LeakyReLU(0.2), ResBlock(vgg_mid_channels),
            ResBlock(vgg_mid_channels),
            nn.Conv2d(vgg_mid_channels, 3, 3, stride=1, padding=1), nn.Tanh())

    def forward(self, input, locations, dictionary):
        """Forward function.

        Args:
            input (Tensor): Input tensor.
            locations (dict): Location dict of facial features.

        Results:
            output (Tensor): Forward output.
        """
        vgg_features = self.vgg_extract(input)
        update_vgg_features = []
        for key in locations.keys():
            if locations[key].sum() == 0:
                return input
            locations[key].to(input.device)
        for i, feature_size in enumerate(self.feature_sizes):
            feature = vgg_features[i]
            update_feature = feature.clone()

            dict_features = dictionary[feature_size]
            left_eye_dict_feature = dict_features['left_eye'].to(input)
            right_eye_dict_feature = dict_features['right_eye'].to(input)
            nose_dict_feature = dict_features['nose'].to(input)
            mouth_dict_feature = dict_features['mouth'].to(input)

            left_eye_location = (locations['left_eye'][0] //
                                 (512 / feature_size)).int()
            right_eye_location = (locations['right_eye'][0] //
                                  (512 / feature_size)).int()
            nose_location = (locations['nose'][0] //
                             (512 / feature_size)).int()
            mouth_location = (locations['mouth'][0] //
                              (512 / feature_size)).int()

            left_eye_feature = feature[:, :, left_eye_location[1]:
                                       left_eye_location[3],
                                       left_eye_location[0]:
                                       left_eye_location[2]].clone()
            right_eye_feature = feature[:, :, right_eye_location[1]:
                                        right_eye_location[3],
                                        right_eye_location[0]:
                                        right_eye_location[2]].clone()
            nose_feature = feature[:, :, nose_location[1]:nose_location[3],
                                   nose_location[0]:nose_location[2]].clone()
            mouth_feature = feature[:, :, mouth_location[1]:mouth_location[3],
                                    mouth_location[0]:mouth_location[2]].clone(
                                    )

            # avoid size=0
            if left_eye_feature.sum() > 0:
                left_eye_feature_resize = F.interpolate(
                    left_eye_feature, (left_eye_dict_feature.size(2),
                                       left_eye_dict_feature.size(3)),
                    mode='bilinear',
                    align_corners=False)
                left_eye_dict_feature_norm = adaptive_instance_normalization(
                    left_eye_dict_feature, left_eye_feature_resize)
                left_eye_score = F.conv2d(left_eye_feature_resize,
                                          left_eye_dict_feature_norm)
                left_eye_score = F.softmax(left_eye_score.view(-1), dim=0)
                left_eye_index = torch.argmax(left_eye_score)
                left_eye_swap = F.interpolate(
                    left_eye_dict_feature_norm[left_eye_index:left_eye_index +
                                               1],
                    (left_eye_feature.size(2), left_eye_feature.size(3)))
                left_eye_block = getattr(self, 'left_eye_' + str(feature_size))
                left_eye_attention = left_eye_block(left_eye_swap -
                                                    left_eye_feature)
                left_eye_attention = left_eye_attention * left_eye_swap
                update_feature[:, :, left_eye_location[1]:left_eye_location[3],
                               left_eye_location[0]:left_eye_location[
                                   2]] = left_eye_attention + left_eye_feature

            if right_eye_feature.sum() > 0:
                right_eye_feature_resize = F.interpolate(
                    right_eye_feature, (right_eye_dict_feature.size(2),
                                        right_eye_dict_feature.size(3)),
                    mode='bilinear',
                    align_corners=False)
                right_eye_dict_feature_norm = adaptive_instance_normalization(
                    right_eye_dict_feature, right_eye_feature_resize)
                right_eye_score = F.conv2d(right_eye_feature_resize,
                                           right_eye_dict_feature_norm)
                right_eye_score = F.softmax(right_eye_score.view(-1), dim=0)
                right_eye_index = torch.argmax(right_eye_score)
                right_eye_swap = F.interpolate(
                    right_eye_dict_feature_norm[
                        right_eye_index:right_eye_index + 1],
                    (right_eye_feature.size(2), right_eye_feature.size(3)))
                right_eye_block = getattr(self,
                                          'right_eye_' + str(feature_size))
                right_eye_attention = right_eye_block(right_eye_swap -
                                                      right_eye_feature)
                right_eye_attention = right_eye_attention * right_eye_swap
                right_eye_feature = right_eye_attention + right_eye_feature
                update_feature[:, :,
                               right_eye_location[1]:right_eye_location[3],
                               right_eye_location[0]:
                               right_eye_location[2]] = right_eye_feature

            if nose_feature.sum() > 0:
                nose_feature_resize = F.interpolate(
                    nose_feature,
                    (nose_dict_feature.size(2), nose_dict_feature.size(3)),
                    mode='bilinear',
                    align_corners=False)
                nose_dict_feature_norm = adaptive_instance_normalization(
                    nose_dict_feature, nose_feature_resize)
                nose_score = F.conv2d(nose_feature_resize,
                                      nose_dict_feature_norm)
                nose_score = F.softmax(nose_score.view(-1), dim=0)
                nose_index = torch.argmax(nose_score)
                nose_swap = F.interpolate(
                    nose_dict_feature_norm[nose_index:nose_index + 1],
                    (nose_feature.size(2), nose_feature.size(3)))
                nose_block = getattr(self, 'nose_' + str(feature_size))
                nose_attention = nose_block(nose_swap - nose_feature)
                nose_attention_feature = nose_attention * nose_swap
                update_feature[:, :, nose_location[1]:nose_location[3],
                               nose_location[0]:nose_location[
                                   2]] = nose_attention_feature + nose_feature

            if mouth_feature.sum() > 0:
                mouth_feature_resize = F.interpolate(
                    mouth_feature,
                    (mouth_dict_feature.size(2), mouth_dict_feature.size(3)),
                    mode='bilinear',
                    align_corners=False)
                mouth_dict_feature_norm = adaptive_instance_normalization(
                    mouth_dict_feature, mouth_feature_resize)
                mouth_score = F.conv2d(mouth_feature_resize,
                                       mouth_dict_feature_norm)
                mouth_score = F.softmax(mouth_score.view(-1), dim=0)
                mouth_index = torch.argmax(mouth_score)
                mouth_swap = F.interpolate(
                    mouth_dict_feature_norm[mouth_index:mouth_index + 1],
                    (mouth_feature.size(2), mouth_feature.size(3)))
                mouth_block = getattr(self, 'mouth_' + str(feature_size))
                mouth_attention = mouth_block(mouth_swap - mouth_feature)
                mouth_attention = mouth_attention * mouth_swap
                update_feature[:, :, mouth_location[1]:mouth_location[3],
                               mouth_location[0]:mouth_location[
                                   2]] = mouth_attention + mouth_feature

            update_vgg_features.append(update_feature)

        feature_vgg = self.ms_dilate(vgg_features[3])
        feature_up0 = self.up0(feature_vgg, update_vgg_features[3])
        feature_up1 = self.up1(feature_up0, update_vgg_features[2])
        feature_up2 = self.up2(feature_up1, update_vgg_features[1])
        feature_up3 = self.up3(feature_up2, update_vgg_features[0])
        output = self.up4(feature_up3)

        return output
