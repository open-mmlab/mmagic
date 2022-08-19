# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model.utils import constant_init, xavier_init
from torch.nn import functional as F


class GCAModule(nn.Module):
    """Guided Contextual Attention Module.

    From https://arxiv.org/pdf/2001.04069.pdf.
    Based on https://github.com/nbei/Deep-Flow-Guided-Video-Inpainting.
    This module use image feature map to augment the alpha feature map with
    guided contextual attention score.

    Image feature and alpha feature are unfolded to small patches and later
    used as conv kernel. Thus, we refer the unfolding size as kernel size.
    Image feature patches have a default kernel size 3 while the kernel size of
    alpha feature patches could be specified by `rate` (see `rate` below). The
    image feature patches are used to convolve with the image feature itself
    to calculate the contextual attention. Then the attention feature map is
    convolved by alpha feature patches to obtain the attention alpha feature.
    At last, the attention alpha feature is added to the input alpha feature.

    Args:
        in_channels (int): Input channels of the guided contextual attention
            module.
        out_channels (int): Output channels of the guided contextual attention
            module.
        kernel_size (int): Kernel size of image feature patches. Default 3.
        stride (int): Stride when unfolding the image feature. Default 1.
        rate (int): The downsample rate of image feature map. The corresponding
            kernel size and stride of alpha feature patches will be `rate x 2`
            and `rate`. It could be regarded as the granularity of the gca
            module. Default: 2.
        pad_args (dict): Parameters of padding when convolve image feature with
            image feature patches or alpha feature patches. Allowed keys are
            `mode` and `value`. See torch.nn.functional.pad() for more
            information. Default: dict(mode='reflect').
        interpolation (str): Interpolation method in upsampling and
            downsampling.
        penalty (float): Punishment hyperparameter to avoid a large correlation
            between each unknown patch and itself.
        eps (float): A small number to avoid dividing by 0 when calculating
            the normed image feature patch. Default: 1e-4.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 rate=2,
                 pad_args=dict(mode='reflect'),
                 interpolation='nearest',
                 penalty=-1e4,
                 eps=1e-4):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.rate = rate
        self.pad_args = pad_args
        self.interpolation = interpolation
        self.penalty = penalty
        self.eps = eps

        # reduced the channels of input image feature.
        self.guidance_conv = nn.Conv2d(in_channels, in_channels // 2, 1)

        # convolution after the attention alpha feature
        self.out_conv = ConvModule(
            out_channels,
            out_channels,
            1,
            norm_cfg=dict(type='BN'),
            act_cfg=None)

        self.init_weights()

    def init_weights(self):
        xavier_init(self.guidance_conv, distribution='uniform')
        xavier_init(self.out_conv.conv, distribution='uniform')
        constant_init(self.out_conv.norm, 1e-3)

    def forward(self, img_feat, alpha_feat, unknown=None, softmax_scale=1.):
        """Forward function of GCAModule.

        Args:
            img_feat (Tensor): Image feature map of shape
                (N, ori_c, ori_h, ori_w).
            alpha_feat (Tensor): Alpha feature map of shape
                (N, alpha_c, ori_h, ori_w).
            unknown (Tensor, optional): Unknown area map generated by trimap.
                If specified, this tensor should have shape
                (N, 1, ori_h, ori_w).
            softmax_scale (float, optional): The softmax scale of the attention
                if unknown area is not provided in forward. Default: 1.

        Returns:
            Tensor: The augmented alpha feature.
        """

        if alpha_feat.shape[2:4] != img_feat.shape[2:4]:
            raise ValueError(
                'image feature size does not align with alpha feature size: '
                f'image feature size {img_feat.shape[2:4]}, '
                f'alpha feature size {alpha_feat.shape[2:4]}')

        if unknown is not None and unknown.shape[2:4] != img_feat.shape[2:4]:
            raise ValueError(
                'image feature size does not align with unknown mask size: '
                f'image feature size {img_feat.shape[2:4]}, '
                f'unknown mask size {unknown.shape[2:4]}')

        # preprocess image feature
        img_feat = self.guidance_conv(img_feat)
        img_feat = F.interpolate(
            img_feat, scale_factor=1 / self.rate, mode=self.interpolation)

        # preprocess unknown mask
        unknown, softmax_scale = self.process_unknown_mask(
            unknown, img_feat, softmax_scale)

        img_ps, alpha_ps, unknown_ps = self.extract_feature_maps_patches(
            img_feat, alpha_feat, unknown)

        # create self correlation mask with shape:
        # (N, img_h*img_w, img_h, img_w)
        self_mask = self.get_self_correlation_mask(img_feat)

        # split tensors by batch dimension; tuple is returned
        img_groups = torch.split(img_feat, 1, dim=0)
        img_ps_groups = torch.split(img_ps, 1, dim=0)
        alpha_ps_groups = torch.split(alpha_ps, 1, dim=0)
        unknown_ps_groups = torch.split(unknown_ps, 1, dim=0)
        scale_groups = torch.split(softmax_scale, 1, dim=0)
        groups = (img_groups, img_ps_groups, alpha_ps_groups,
                  unknown_ps_groups, scale_groups)

        out = []
        # i is the virtual index of the sample in the current batch
        for img_i, img_ps_i, alpha_ps_i, unknown_ps_i, scale_i in zip(*groups):
            similarity_map = self.compute_similarity_map(img_i, img_ps_i)

            gca_score = self.compute_guided_attention_score(
                similarity_map, unknown_ps_i, scale_i, self_mask)

            out_i = self.propagate_alpha_feature(gca_score, alpha_ps_i)

            out.append(out_i)

        out = torch.cat(out, dim=0)
        out.reshape_as(alpha_feat)

        out = self.out_conv(out) + alpha_feat
        return out

    def extract_feature_maps_patches(self, img_feat, alpha_feat, unknown):
        """Extract image feature, alpha feature unknown patches.

        Args:
            img_feat (Tensor): Image feature map of shape
                (N, img_c, img_h, img_w).
            alpha_feat (Tensor): Alpha feature map of shape
                (N, alpha_c, ori_h, ori_w).
            unknown (Tensor, optional): Unknown area map generated by trimap of
                shape (N, 1, img_h, img_w).

        Returns:
            tuple: 3-tuple of

                ``Tensor``: Image feature patches of shape \
                    (N, img_h*img_w, img_c, img_ks, img_ks).

                ``Tensor``: Guided contextual attention alpha feature map. \
                    (N, img_h*img_w, alpha_c, alpha_ks, alpha_ks).

                ``Tensor``: Unknown mask of shape (N, img_h*img_w, 1, 1).
        """
        # extract image feature patches with shape:
        # (N, img_h*img_w, img_c, img_ks, img_ks)
        img_ks = self.kernel_size
        img_ps = self.extract_patches(img_feat, img_ks, self.stride)

        # extract alpha feature patches with shape:
        # (N, img_h*img_w, alpha_c, alpha_ks, alpha_ks)
        alpha_ps = self.extract_patches(alpha_feat, self.rate * 2, self.rate)

        # extract unknown mask patches with shape: (N, img_h*img_w, 1, 1)
        unknown_ps = self.extract_patches(unknown, img_ks, self.stride)
        unknown_ps = unknown_ps.squeeze(dim=2)  # squeeze channel dimension
        unknown_ps = unknown_ps.mean(dim=[2, 3], keepdim=True)

        return img_ps, alpha_ps, unknown_ps

    def compute_similarity_map(self, img_feat, img_ps):
        """Compute similarity between image feature patches.

        Args:
            img_feat (Tensor): Image feature map of shape
                (1, img_c, img_h, img_w).
            img_ps (Tensor): Image feature patches tensor of shape
                (1, img_h*img_w, img_c, img_ks, img_ks).

        Returns:
            Tensor: Similarity map between image feature patches with shape \
                (1, img_h*img_w, img_h, img_w).
        """
        img_ps = img_ps[0]  # squeeze dim 0
        # convolve the feature to get correlation (similarity) map
        escape_NaN = torch.FloatTensor([self.eps]).to(img_feat)
        img_ps_normed = img_ps / torch.max(self.l2_norm(img_ps), escape_NaN)
        img_feat = self.pad(img_feat, self.kernel_size, self.stride)
        similarity_map = F.conv2d(img_feat, img_ps_normed)

        return similarity_map

    def compute_guided_attention_score(self, similarity_map, unknown_ps, scale,
                                       self_mask):
        """Compute guided attention score.

        Args:
            similarity_map (Tensor): Similarity map of image feature with shape
                (1, img_h*img_w, img_h, img_w).
            unknown_ps (Tensor): Unknown area patches tensor of shape
                (1, img_h*img_w, 1, 1).
            scale (Tensor): Softmax scale of known and unknown area:
                [unknown_scale, known_scale].
            self_mask (Tensor): Self correlation mask of shape
                (1, img_h*img_w, img_h, img_w). At (1, i*i, i, i) mask value
                equals -1e4 for i in [1, img_h*img_w] and other area is all
                zero.

        Returns:
            Tensor: Similarity map between image feature patches with shape \
                (1, img_h*img_w, img_h, img_w).
        """
        # scale the correlation with predicted scale factor for known and
        # unknown area
        unknown_scale, known_scale = scale[0]
        out = similarity_map * (
            unknown_scale * unknown_ps.gt(0.).float() +
            known_scale * unknown_ps.le(0.).float())
        # mask itself, self-mask only applied to unknown area
        out = out + self_mask * unknown_ps
        gca_score = F.softmax(out, dim=1)

        return gca_score

    def propagate_alpha_feature(self, gca_score, alpha_ps):
        """Propagate alpha feature based on guided attention score.

        Args:
            gca_score (Tensor): Guided attention score map of shape
                (1, img_h*img_w, img_h, img_w).
            alpha_ps (Tensor): Alpha feature patches tensor of shape
                (1, img_h*img_w, alpha_c, alpha_ks, alpha_ks).

        Returns:
            Tensor: Propagated alpha feature map of shape \
                (1, alpha_c, alpha_h, alpha_w).
        """
        alpha_ps = alpha_ps[0]  # squeeze dim 0
        if self.rate == 1:
            gca_score = self.pad(gca_score, kernel_size=2, stride=1)
            alpha_ps = alpha_ps.permute(1, 0, 2, 3)
            out = F.conv2d(gca_score, alpha_ps) / 4.
        else:
            out = F.conv_transpose2d(
                gca_score, alpha_ps, stride=self.rate, padding=1) / 4.

        return out

    def process_unknown_mask(self, unknown, img_feat, softmax_scale):
        """Process unknown mask.

        Args:
            unknown (Tensor, optional): Unknown area map generated by trimap of
                shape (N, 1, ori_h, ori_w)
            img_feat (Tensor): The interpolated image feature map of shape
                (N, img_c, img_h, img_w).
            softmax_scale (float, optional): The softmax scale of the attention
                if unknown area is not provided in forward. Default: 1.

        Returns:
            tuple: 2-tuple of

                ``Tensor``: Interpolated unknown area map of shape \
                    (N, img_h*img_w, img_h, img_w).

                ``Tensor``: Softmax scale tensor of known and unknown area of \
                    shape (N, 2).
        """
        n, _, h, w = img_feat.shape

        if unknown is not None:
            unknown = unknown.clone()
            unknown = F.interpolate(
                unknown, scale_factor=1 / self.rate, mode=self.interpolation)
            unknown_mean = unknown.mean(dim=[2, 3])
            known_mean = 1 - unknown_mean
            unknown_scale = torch.clamp(
                torch.sqrt(unknown_mean / known_mean), 0.1, 10).to(img_feat)
            known_scale = torch.clamp(
                torch.sqrt(known_mean / unknown_mean), 0.1, 10).to(img_feat)
            softmax_scale = torch.cat([unknown_scale, known_scale], dim=1)
        else:
            unknown = torch.ones((n, 1, h, w)).to(img_feat)
            softmax_scale = torch.FloatTensor(
                [softmax_scale,
                 softmax_scale]).view(1, 2).repeat(n, 1).to(img_feat)

        return unknown, softmax_scale

    def extract_patches(self, x, kernel_size, stride):
        """Extract feature patches.

        The feature map will be padded automatically to make sure the number of
        patches is equal to `(H / stride) * (W / stride)`.

        Args:
            x (Tensor): Feature map of shape (N, C, H, W).
            kernel_size (int): Size of each patches.
            stride (int): Stride between patches.

        Returns:
            Tensor: Extracted patches of shape \
                (N, (H / stride) * (W / stride) , C, kernel_size, kernel_size).
        """
        n, c, _, _ = x.shape
        x = self.pad(x, kernel_size, stride)
        x = F.unfold(x, (kernel_size, kernel_size), stride=(stride, stride))
        x = x.permute(0, 2, 1)
        x = x.reshape(n, -1, c, kernel_size, kernel_size)
        return x

    def pad(self, x, kernel_size, stride):
        left = (kernel_size - stride + 1) // 2
        right = (kernel_size - stride) // 2
        pad = (left, right, left, right)
        return F.pad(x, pad, **self.pad_args)

    def get_self_correlation_mask(self, img_feat):
        _, _, h, w = img_feat.shape
        # As ONNX does not support dynamic num_classes, we have to convert it
        # into an integer
        self_mask = F.one_hot(
            torch.arange(h * w).view(h, w), num_classes=int(h * w))
        self_mask = self_mask.permute(2, 0, 1).view(1, h * w, h, w)
        # use large negative value to mask out self-correlation before softmax
        self_mask = self_mask * self.penalty
        return self_mask.to(img_feat)

    @staticmethod
    def l2_norm(x):
        x = x**2
        x = x.sum(dim=[1, 2, 3], keepdim=True)
        return torch.sqrt(x)
