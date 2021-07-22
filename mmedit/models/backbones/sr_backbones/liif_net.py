import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import load_checkpoint

from mmedit.datasets.pipelines.utils import make_coord
from mmedit.models.builder import build_backbone, build_component
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger


class LIIFNet(nn.Module):
    """LIIF net for single image super-resolution, CVPR, 2021.

    Paper: Learning Continuous Image Representation with
           Local Implicit Image Function

    The subclasses should define `generator` with `encoder` and `imnet`,
        and overwrite the function `gen_feature`.
    If `encoder` does not contain `mid_channels`, `__init__` should be
        overwrite.

    Args:
        encoder (dict): Config for the generator.
        imnet (dict): Config for the imnet.
        local_ensemble (bool): Whether to use local ensemble. Default: True.
        feat_unfold (bool): Whether to use feature unfold. Default: True.
        cell_decode (bool): Whether to use cell decode. Default: True.
        eval_bsize (int): Size of batched predict. Default: None.
    """

    def __init__(self,
                 encoder,
                 imnet,
                 local_ensemble=True,
                 feat_unfold=True,
                 cell_decode=True,
                 eval_bsize=None):
        super().__init__()

        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        self.eval_bsize = eval_bsize

        # model
        self.encoder = build_backbone(encoder)
        imnet_in_dim = self.encoder.mid_channels
        if self.feat_unfold:
            imnet_in_dim *= 9
        imnet_in_dim += 2  # attach coordinates
        if self.cell_decode:
            imnet_in_dim += 2
        imnet['in_dim'] = imnet_in_dim
        self.imnet = build_component(imnet)

    def forward(self, x, coord, cell, test_mode=False):
        """Forward function.

        Args:
            x: input tensor.
            coord (Tensor): coordinates tensor.
            cell (Tensor): cell tensor.
            test_mode (bool): Whether in test mode or not. Default: False.

        Returns:
            pred (Tensor): output of model.
        """

        feature = self.gen_feature(x)
        if self.eval_bsize is None or not test_mode:
            pred = self.query_rgb(feature, coord, cell)
        else:
            pred = self.batched_predict(feature, coord, cell)

        return pred

    def query_rgb(self, feature, coord, cell=None):
        """Query RGB value of GT.

        Adapted from 'https://github.com/yinboc/liif.git'
        'liif/models/liif.py'
        Copyright (c) 2020, Yinbo Chen, under BSD 3-Clause License.

        Args:
            feature (Tensor): encoded feature.
            coord (Tensor): coord tensor, shape (BHW, 2).
            cell (Tensor | None): cell tensor. Default: None.

        Returns:
            result (Tensor): (part of) output.
        """

        if self.imnet is None:
            result = F.grid_sample(
                feature,
                coord.flip(-1).unsqueeze(1),
                mode='nearest',
                align_corners=False)
            result = result[:, :, 0, :].permute(0, 2, 1)
            return result

        if self.feat_unfold:
            feature = F.unfold(
                feature, 3,
                padding=1).view(feature.shape[0], feature.shape[1] * 9,
                                feature.shape[2], feature.shape[3])

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        radius_x = 2 / feature.shape[-2] / 2
        radius_y = 2 / feature.shape[-1] / 2

        feat_coord = make_coord(feature.shape[-2:], flatten=False) \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feature.shape[0], 2, *feature.shape[-2:])
        feat_coord = feat_coord.to(coord)

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * radius_x + eps_shift
                coord_[:, :, 1] += vy * radius_y + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                query_feat = F.grid_sample(
                    feature, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                query_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - query_coord
                rel_coord[:, :, 0] *= feature.shape[-2]
                rel_coord[:, :, 1] *= feature.shape[-1]
                mid_tensor = torch.cat([query_feat, rel_coord], dim=-1)

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feature.shape[-2]
                    rel_cell[:, :, 1] *= feature.shape[-1]
                    mid_tensor = torch.cat([mid_tensor, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(mid_tensor.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        total_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            areas = areas[::-1]
        result = 0
        for pred, area in zip(preds, areas):
            result = result + pred * (area / total_area).unsqueeze(-1)

        return result

    def batched_predict(self, x, coord, cell):
        """Batched predict.

        Args:
            x (Tensor): Input tensor.
            coord (Tensor): coord tensor.
            cell (Tensor): cell tensor.

        Returns:
            pred (Tensor): output of model.
        """
        with torch.no_grad():
            n = coord.shape[1]
            left = 0
            preds = []
            while left < n:
                right = min(left + self.eval_bsize, n)
                pred = self.query_rgb(x, coord[:, left:right, :],
                                      cell[:, left:right, :])
                preds.append(pred)
                left = right
            pred = torch.cat(preds, dim=1)
        return pred

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
        elif pretrained is not None:
            raise TypeError('"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')


@BACKBONES.register_module()
class LIIFEDSR(LIIFNet):
    """LIIF net based on EDSR.

    Paper: Learning Continuous Image Representation with
           Local Implicit Image Function

    Args:
        encoder (dict): Config for the generator.
        imnet (dict): Config for the imnet.
        local_ensemble (bool): Whether to use local ensemble. Default: True.
        feat_unfold (bool): Whether to use feature unfold. Default: True.
        cell_decode (bool): Whether to use cell decode. Default: True.
        eval_bsize (int): Size of batched predict. Default: None.
    """

    def __init__(self,
                 encoder,
                 imnet,
                 local_ensemble=True,
                 feat_unfold=True,
                 cell_decode=True,
                 eval_bsize=None):
        super().__init__(
            encoder=encoder,
            imnet=imnet,
            local_ensemble=local_ensemble,
            feat_unfold=feat_unfold,
            cell_decode=cell_decode,
            eval_bsize=eval_bsize)

        self.conv_first = self.encoder.conv_first
        self.body = self.encoder.body
        self.conv_after_body = self.encoder.conv_after_body
        del self.encoder

    def gen_feature(self, x):
        """Generate feature.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        x = self.conv_first(x)
        res = self.body(x)
        res = self.conv_after_body(res)
        res += x

        return res


@BACKBONES.register_module()
class LIIFRDN(LIIFNet):
    """LIIF net based on RDN.

    Paper: Learning Continuous Image Representation with
           Local Implicit Image Function

    Args:
        encoder (dict): Config for the generator.
        imnet (dict): Config for the imnet.
        local_ensemble (bool): Whether to use local ensemble. Default: True.
        feat_unfold (bool): Whether to use feat unfold. Default: True.
        cell_decode (bool): Whether to use cell decode. Default: True.
        eval_bsize (int): Size of batched predict. Default: None.
    """

    def __init__(self,
                 encoder,
                 imnet,
                 local_ensemble=True,
                 feat_unfold=True,
                 cell_decode=True,
                 eval_bsize=None):
        super().__init__(
            encoder=encoder,
            imnet=imnet,
            local_ensemble=local_ensemble,
            feat_unfold=feat_unfold,
            cell_decode=cell_decode,
            eval_bsize=eval_bsize)

        self.sfe1 = self.encoder.sfe1
        self.sfe2 = self.encoder.sfe2
        self.rdbs = self.encoder.rdbs
        self.gff = self.encoder.gff
        self.num_blocks = self.encoder.num_blocks
        del self.encoder

    def gen_feature(self, x):
        """Generate feature.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.num_blocks):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) + sfe1

        return x
