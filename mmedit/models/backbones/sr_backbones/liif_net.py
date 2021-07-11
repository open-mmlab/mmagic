import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import load_checkpoint

from mmedit.datasets.pipelines.utils import make_coord
from mmedit.models.builder import build_backbone, build_component
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger


class LIIFNet(nn.Module):
    """LIIF net for single image super-resolution.

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
        imnet_in_dim += 2  # attach coord
        if self.cell_decode:
            imnet_in_dim += 2
        imnet['in_dim'] = imnet_in_dim
        self.imnet = build_component(imnet)

    def forward(self, x, coord, cell, test_mode=False):
        """Forward function.

        Args:
            x: input tensor.
            coord (Tensor): coord tensor.
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
        feat = feature

        if self.imnet is None:
            result = F.grid_sample(
                feat,
                coord.flip(-1).unsqueeze(1),
                mode='nearest',
                align_corners=False)
            result = result[:, :, 0, :].permute(0, 2, 1)
            return result

        if self.feat_unfold:
            feat = F.unfold(
                feat, 3, padding=1).view(feat.shape[0], feat.shape[1] * 9,
                                         feat.shape[2], feat.shape[3])

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False) \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])
        feat_coord = feat_coord.to(coord)

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                query_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                query_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - query_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                mid_tensor = torch.cat([query_feat, rel_coord], dim=-1)

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    mid_tensor = torch.cat([mid_tensor, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(mid_tensor.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]
            areas[0] = areas[3]
            areas[3] = t
            t = areas[1]
            areas[1] = areas[2]
            areas[2] = t
        result = 0
        for pred, area in zip(preds, areas):
            result = result + pred * (area / tot_area).unsqueeze(-1)

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
            ql = 0
            preds = []
            while ql < n:
                qr = min(ql + self.eval_bsize, n)
                pred = self.query_rgb(x, coord[:, ql:qr, :], cell[:, ql:qr, :])
                preds.append(pred)
                ql = qr
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
