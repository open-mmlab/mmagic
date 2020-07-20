import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from mmcv.runner import load_checkpoint
from torch.nn.modules.utils import _pair

from mmedit.models.common import (PixelShufflePack, ResidualBlockNoBN,
                                  make_layer)
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger


class ModulatedDCNPack(ModulatedDeformConv2d):
    """Modulated Deformable Convolutional Pack.

    Different from the official DCN, which generates offsets and masks from
    the preceding features, this ModulatedDCNPack takes another different
    feature to generate masks and offsets.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    def __init__(self, *args, **kwargs):
        super(ModulatedDCNPack, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            bias=True)
        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset, val=0, bias=0)

    def forward(self, x, extra_feat):
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)


class PCDAlignment(nn.Module):
    """Alignment module using Pyramid, Cascading and Deformable convolution
    (PCD). It is used in EDVRNet.

    Args:
        mid_channels (int): Number of the channels of middle features.
            Default: 64.
        deform_groups (int): Deformable groups. Defaults: 8.
        act_cfg (dict): Activation function config for ConvModule.
            Default: LeakyReLU with negative_slope=0.1.
    """

    def __init__(self,
                 mid_channels=64,
                 deform_groups=8,
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        super(PCDAlignment, self).__init__()

        # Pyramid has three levels:
        # L3: level 3, 1/4 spatial size
        # L2: level 2, 1/2 spatial size
        # L1: level 1, original spatial size
        self.offset_conv1 = nn.ModuleDict()
        self.offset_conv2 = nn.ModuleDict()
        self.offset_conv3 = nn.ModuleDict()
        self.dcn_pack = nn.ModuleDict()
        self.feat_conv = nn.ModuleDict()
        for i in range(3, 0, -1):
            level = f'l{i}'
            self.offset_conv1[level] = ConvModule(
                mid_channels * 2, mid_channels, 3, padding=1, act_cfg=act_cfg)
            if i == 3:
                self.offset_conv2[level] = ConvModule(
                    mid_channels, mid_channels, 3, padding=1, act_cfg=act_cfg)
            else:
                self.offset_conv2[level] = ConvModule(
                    mid_channels * 2,
                    mid_channels,
                    3,
                    padding=1,
                    act_cfg=act_cfg)
                self.offset_conv3[level] = ConvModule(
                    mid_channels, mid_channels, 3, padding=1, act_cfg=act_cfg)
            self.dcn_pack[level] = ModulatedDCNPack(
                mid_channels,
                mid_channels,
                3,
                padding=1,
                deform_groups=deform_groups)

            if i < 3:
                act_cfg_ = act_cfg if i == 2 else None
                self.feat_conv[level] = ConvModule(
                    mid_channels * 2,
                    mid_channels,
                    3,
                    padding=1,
                    act_cfg=act_cfg_)

        # Cascading DCN
        self.cas_offset_conv1 = ConvModule(
            mid_channels * 2, mid_channels, 3, padding=1, act_cfg=act_cfg)
        self.cas_offset_conv2 = ConvModule(
            mid_channels, mid_channels, 3, padding=1, act_cfg=act_cfg)
        self.cas_dcnpack = ModulatedDCNPack(
            mid_channels,
            mid_channels,
            3,
            padding=1,
            deform_groups=deform_groups)

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, neighbor_feats, ref_feats):
        """Forward function for PCDAlignment.

        Align neighboring frames to the reference frame in the feature level.

        Args:
            neighbor_feats (list[Tensor]): List of neighboring features. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (n, c, h, w).
            ref_feats (list[Tensor]): List of reference features. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (n, c, h, w).

        Returns:
            Tensor: Aligned features.
        """
        # The number of pyramid levels is 3.
        assert len(neighbor_feats) == 3 and len(ref_feats) == 3, (
            'The length of neighbor_feats and ref_feats must be both 3, '
            f'but got {len(neighbor_feats)} and {len(ref_feats)}')

        # Pyramids
        upsampled_offset, upsampled_feat = None, None
        for i in range(3, 0, -1):
            level = f'l{i}'
            offset = torch.cat([neighbor_feats[i - 1], ref_feats[i - 1]],
                               dim=1)
            offset = self.offset_conv1[level](offset)
            if i == 3:
                offset = self.offset_conv2[level](offset)
            else:
                offset = self.offset_conv2[level](
                    torch.cat([offset, upsampled_offset], dim=1))
                offset = self.offset_conv3[level](offset)

            feat = self.dcn_pack[level](neighbor_feats[i - 1], offset)
            if i == 3:
                feat = self.lrelu(feat)
            else:
                feat = self.feat_conv[level](
                    torch.cat([feat, upsampled_feat], dim=1))

            if i > 1:
                # upsample offset and features
                upsampled_offset = self.upsample(offset) * 2
                upsampled_feat = self.upsample(feat)

        # Cascading
        offset = torch.cat([feat, ref_feats[0]], dim=1)
        offset = self.cas_offset_conv2(self.cas_offset_conv1(offset))
        feat = self.lrelu(self.cas_dcnpack(feat, offset))
        return feat


class TSAFusion(nn.Module):
    """Temporal Spatial Attention (TSA) fusion module. It is used in EDVRNet.

    Args:
        mid_channels (int): Number of the channels of middle features.
            Default: 64.
        num_frames (int): Number of frames. Default: 5.
        center_frame_idx (int): The index of center frame. Default: 2.
        act_cfg (dict): Activation function config for ConvModule.
            Default: LeakyReLU with negative_slope=0.1.
    """

    def __init__(self,
                 mid_channels=64,
                 num_frames=5,
                 center_frame_idx=2,
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        super(TSAFusion, self).__init__()
        self.center_frame_idx = center_frame_idx
        # temporal attention (before fusion conv)
        self.temporal_attn1 = nn.Conv2d(
            mid_channels, mid_channels, 3, padding=1)
        self.temporal_attn2 = nn.Conv2d(
            mid_channels, mid_channels, 3, padding=1)
        self.feat_fusion = ConvModule(
            num_frames * mid_channels, mid_channels, 1, act_cfg=act_cfg)

        # spatial attention (after fusion conv)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.spatial_attn1 = ConvModule(
            num_frames * mid_channels, mid_channels, 1, act_cfg=act_cfg)
        self.spatial_attn2 = ConvModule(
            mid_channels * 2, mid_channels, 1, act_cfg=act_cfg)
        self.spatial_attn3 = ConvModule(
            mid_channels, mid_channels, 3, padding=1, act_cfg=act_cfg)
        self.spatial_attn4 = ConvModule(
            mid_channels, mid_channels, 1, act_cfg=act_cfg)
        self.spatial_attn5 = nn.Conv2d(
            mid_channels, mid_channels, 3, padding=1)
        self.spatial_attn_l1 = ConvModule(
            mid_channels, mid_channels, 1, act_cfg=act_cfg)
        self.spatial_attn_l2 = ConvModule(
            mid_channels * 2, mid_channels, 3, padding=1, act_cfg=act_cfg)
        self.spatial_attn_l3 = ConvModule(
            mid_channels, mid_channels, 3, padding=1, act_cfg=act_cfg)
        self.spatial_attn_add1 = ConvModule(
            mid_channels, mid_channels, 1, act_cfg=act_cfg)
        self.spatial_attn_add2 = nn.Conv2d(mid_channels, mid_channels, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, aligned_feat):
        """Forward function for TSAFusion.

        Args:
            aligned_feat (Tensor): Aligned features with shape (n, t, c, h, w).

        Returns:
            Tensor: Features after TSA with the shape (n, c, h, w).
        """
        n, t, c, h, w = aligned_feat.size()
        # temporal attention
        embedding_ref = self.temporal_attn1(
            aligned_feat[:, self.center_frame_idx, :, :, :].clone())
        emb = self.temporal_attn2(aligned_feat.view(-1, c, h, w))
        emb = emb.view(n, t, -1, h, w)  # (n, t, c, h, w)

        corr_l = []  # correlation list
        for i in range(t):
            emb_neighbor = emb[:, i, :, :, :]
            corr = torch.sum(emb_neighbor * embedding_ref, 1)  # (n, h, w)
            corr_l.append(corr.unsqueeze(1))  # (n, 1, h, w)
        corr_prob = torch.sigmoid(torch.cat(corr_l, dim=1))  # (n, t, h, w)
        corr_prob = corr_prob.unsqueeze(2).expand(n, t, c, h, w)
        corr_prob = corr_prob.contiguous().view(n, -1, h, w)  # (n, t*c, h, w)
        aligned_feat = aligned_feat.view(n, -1, h, w) * corr_prob

        # fusion
        feat = self.feat_fusion(aligned_feat)

        # spatial attention
        attn = self.spatial_attn1(aligned_feat)
        attn_max = self.max_pool(attn)
        attn_avg = self.avg_pool(attn)
        attn = self.spatial_attn2(torch.cat([attn_max, attn_avg], dim=1))
        # pyramid levels
        attn_level = self.spatial_attn_l1(attn)
        attn_max = self.max_pool(attn_level)
        attn_avg = self.avg_pool(attn_level)
        attn_level = self.spatial_attn_l2(
            torch.cat([attn_max, attn_avg], dim=1))
        attn_level = self.spatial_attn_l3(attn_level)
        attn_level = self.upsample(attn_level)

        attn = self.spatial_attn3(attn) + attn_level
        attn = self.spatial_attn4(attn)
        attn = self.upsample(attn)
        attn = self.spatial_attn5(attn)
        attn_add = self.spatial_attn_add2(self.spatial_attn_add1(attn))
        attn = torch.sigmoid(attn)

        # after initialization, * 2 makes (attn * 2) to be close to 1.
        feat = feat * attn * 2 + attn_add
        return feat


@BACKBONES.register_module()
class EDVRNet(nn.Module):
    """EDVR network structure for video super-resolution.

    Now only support X4 upsampling factor.
    Paper:
    EDVR: Video Restoration with Enhanced Deformable Convolutional Networks.

    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        num_frames (int): Number of input frames. Default: 5.
        deform_groups (int): Deformable groups. Defaults: 8.
        num_blocks_extraction (int): Number of blocks for feature extraction.
            Default: 5.
        num_blocks_reconstruction (int): Number of blocks for reconstruction.
            Default: 10.
        center_frame_idx (int): The index of center frame. Frame counting from
            0. Default: 2.
        with_tsa (bool): Whether to use TSA module. Default: True.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=64,
                 num_frames=5,
                 deform_groups=8,
                 num_blocks_extraction=5,
                 num_blocks_reconstruction=10,
                 center_frame_idx=2,
                 with_tsa=True):
        super(EDVRNet, self).__init__()
        self.center_frame_idx = center_frame_idx
        self.with_tsa = with_tsa
        act_cfg = dict(type='LeakyReLU', negative_slope=0.1)

        self.conv_first = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.feature_extraction = make_layer(
            ResidualBlockNoBN,
            num_blocks_extraction,
            mid_channels=mid_channels)

        # generate pyramid features
        self.feat_l2_conv1 = ConvModule(
            mid_channels, mid_channels, 3, 2, 1, act_cfg=act_cfg)
        self.feat_l2_conv2 = ConvModule(
            mid_channels, mid_channels, 3, 1, 1, act_cfg=act_cfg)
        self.feat_l3_conv1 = ConvModule(
            mid_channels, mid_channels, 3, 2, 1, act_cfg=act_cfg)
        self.feat_l3_conv2 = ConvModule(
            mid_channels, mid_channels, 3, 1, 1, act_cfg=act_cfg)
        # pcd alignment
        self.pcd_alignment = PCDAlignment(
            mid_channels=mid_channels, deform_groups=deform_groups)
        # fusion
        if self.with_tsa:
            self.fusion = TSAFusion(
                mid_channels=mid_channels,
                num_frames=num_frames,
                center_frame_idx=self.center_frame_idx)
        else:
            self.fusion = nn.Conv2d(num_frames * mid_channels, mid_channels, 1,
                                    1)

        # reconstruction
        self.reconstruction = make_layer(
            ResidualBlockNoBN,
            num_blocks_reconstruction,
            mid_channels=mid_channels)
        # upsample
        self.upsample1 = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(
            mid_channels, 64, 2, upsample_kernel=3)
        # we fix the output channels in the last few layers to 64.
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=False)
        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        """Forward function for EDVRNet.

        Args:
            x (Tensor): Input tensor with shape (n, t, c, h, w).

        Returns:
            Tensor: SR center frame with shape (n, c, h, w).
        """
        n, t, c, h, w = x.size()
        assert h % 4 == 0 and w % 4 == 0, (
            'The height and width of inputs should be a multiple of 4, '
            f'but got {h} and {w}.')

        x_center = x[:, self.center_frame_idx, :, :, :].contiguous()

        # extract LR features
        # L1
        l1_feat = self.lrelu(self.conv_first(x.view(-1, c, h, w)))
        l1_feat = self.feature_extraction(l1_feat)
        # L2
        l2_feat = self.feat_l2_conv2(self.feat_l2_conv1(l1_feat))
        # L3
        l3_feat = self.feat_l3_conv2(self.feat_l3_conv1(l2_feat))

        l1_feat = l1_feat.view(n, t, -1, h, w)
        l2_feat = l2_feat.view(n, t, -1, h // 2, w // 2)
        l3_feat = l3_feat.view(n, t, -1, h // 4, w // 4)

        # pcd alignment
        ref_feats = [  # reference feature list
            l1_feat[:, self.center_frame_idx, :, :, :].clone(),
            l2_feat[:, self.center_frame_idx, :, :, :].clone(),
            l3_feat[:, self.center_frame_idx, :, :, :].clone()
        ]
        aligned_feat = []
        for i in range(t):
            neighbor_feats = [
                l1_feat[:, i, :, :, :].clone(), l2_feat[:, i, :, :, :].clone(),
                l3_feat[:, i, :, :, :].clone()
            ]
            aligned_feat.append(self.pcd_alignment(neighbor_feats, ref_feats))
        aligned_feat = torch.stack(aligned_feat, dim=1)  # (n, t, c, h, w)

        if self.with_tsa:
            feat = self.fusion(aligned_feat)
        else:
            aligned_feat = aligned_feat.view(n, -1, h, w)
            feat = self.fusion(aligned_feat)

        # reconstruction
        out = self.reconstruction(feat)
        out = self.lrelu(self.upsample1(out))
        out = self.lrelu(self.upsample2(out))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)
        base = self.img_upsample(x_center)
        out += base
        return out

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is None:
            if self.with_tsa:
                for module in [
                        self.fusion.feat_fusion, self.fusion.spatial_attn1,
                        self.fusion.spatial_attn2, self.fusion.spatial_attn3,
                        self.fusion.spatial_attn4, self.fusion.spatial_attn_l1,
                        self.fusion.spatial_attn_l2,
                        self.fusion.spatial_attn_l3,
                        self.fusion.spatial_attn_add1
                ]:
                    kaiming_init(
                        module.conv,
                        a=0.1,
                        mode='fan_out',
                        nonlinearity='leaky_relu',
                        bias=0,
                        distribution='uniform')
        else:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')
