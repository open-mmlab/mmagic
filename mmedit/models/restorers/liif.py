import math
import numbers
import os.path as osp

import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16, load_checkpoint

from mmedit.core import tensor2img
from mmedit.utils import get_root_logger
from ...datasets.pipelines.utils import make_coord
from ..builder import build_backbone, build_component, build_loss
from ..registry import MODELS
from .basic_restorer import BasicRestorer


@MODELS.register_module()
class LIIF(BasicRestorer):
    """LIIF model for single image super-resolution.

    Paper: Learning Continuous Image Representation with
           Local Implicit Image Function

    Args:
        generator (dict): Config for the generator.
        imnet (dict): Config for the imnet.
        local_ensemble (bool): Whether to use local ensemble.
            Default: True.
        feat_unfold (bool): Whether to use feat unfold. Default: True.
        cell_decode (bool): Whether to use cell decode. Default: True.
        rgb_mean (tuple[float]): Data mean.
            Default: (0.5, 0.5, 0.5).
        rgb_std (tuple[float]): Data std.
            Default: (0.5, 0.5, 0.5).
        eval_bsize (int): Size of batched predict. Default: None.
        pixel_loss (dict): Config for the pixel loss. Default: None.
        train_cfg (dict): Config for train. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path for pretrained model. Default: None.
    """

    def __init__(self,
                 generator,
                 imnet,
                 local_ensemble=True,
                 feat_unfold=True,
                 cell_decode=True,
                 rgb_mean=(0.5, 0.5, 0.5),
                 rgb_std=(0.5, 0.5, 0.5),
                 eval_bsize=None,
                 pixel_loss=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):

        super(BasicRestorer, self).__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        self.eval_bsize = eval_bsize
        self.feat = None

        # norm
        rgb_mean = torch.FloatTensor(rgb_mean)
        rgb_std = torch.FloatTensor(rgb_std)
        self.lq_mean = rgb_mean.view(1, -1, 1, 1)
        self.lq_std = rgb_std.view(1, -1, 1, 1)
        self.gt_mean = rgb_mean.view(1, 1, -1)
        self.gt_std = rgb_std.view(1, 1, -1)

        # model
        generator_model = build_backbone(generator)
        generator_list = list(generator_model.children())
        self.head = generator_list[0]
        self.encoder = nn.Sequential(*generator_list[1:-2])
        imnet_in_dim = generator_model.mid_channels
        if self.feat_unfold:
            imnet_in_dim *= 9
        imnet_in_dim += 2  # attach coord
        if self.cell_decode:
            imnet_in_dim += 2
        imnet['in_dim'] = imnet_in_dim
        self.imnet = build_component(imnet)

        # support fp16
        self.fp16_enabled = False

        # loss
        self.pixel_loss = build_loss(pixel_loss) if pixel_loss else None

        self.init_weights(pretrained)

    def train_step(self, data_batch, optimizer):
        """Train step.

        Args:
            data_batch (dict): A batch of data, which requires
                'coord', 'lq', 'gt', 'cell'
            optimizer (obj): Optimizer.

        Returns:
            dict: Returned output, which includes:
                log_vars, num_samples, results (lq, gt and preq).

        """
        # data
        coord = data_batch['coord']
        cell = data_batch['cell']
        lq = data_batch['lq']
        gt = data_batch['gt']

        # norm
        self.lq_mean = self.lq_mean.to(lq)
        self.lq_std = self.lq_std.to(lq)
        self.gt_mean = self.gt_mean.to(gt)
        self.gt_std = self.gt_std.to(gt)
        lq = (lq - self.lq_mean) / self.lq_std
        gt = (gt - self.gt_mean) / self.gt_std

        # generator
        self.gen_feat(lq)
        preq = self.query_rgb(coord, cell)

        # loss
        losses = dict()
        log_vars = dict()
        losses['loss_pix'] = self.pixel_loss(preq, gt)

        # parse loss
        loss, log_vars = self.parse_losses(losses)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log_vars.pop('loss')  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), gt=gt.cpu(), output=preq.cpu()))

        preq = None
        loss = None

        return outputs

    def forward_test(self,
                     lq,
                     gt,
                     coord,
                     cell,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        """Testing forward function.

        Args:
            lq (Tensor): LQ image.
            gt (Tensor): GT image.
            coord (Tensor): Coord tensor.
            cell (Tensor): Cell tensor.
            meta (list[dict]): Meta data, such as path of GT file.
                Default: None.
            save_image (bool): Whether to save image. Default: False.
            save_path (str): Path to save image. Default: None.
            iteration (int): Iteration for the saving image name.
                Default: None.

        Returns:
            dict: Output results, which contain either key(s)
                1. 'eval_result'.
                2. 'lq', 'pred'.
                3. 'lq', 'pred', 'gt'.
        """
        # norm
        self.lq_mean = self.lq_mean.to(lq)
        self.lq_std = self.lq_std.to(lq)
        lq = (lq - self.lq_mean) / self.lq_std

        # generator
        with torch.no_grad():
            self.gen_feat(lq)
            if self.eval_bsize is None:
                pred = self.query_rgb(coord, cell)
            else:
                pred = self.batched_predict(coord, cell)
            self.gt_mean = self.gt_mean.to(pred)
            self.gt_std = self.gt_std.to(pred)
            pred = pred * self.gt_std + self.gt_mean
            pred.clamp_(0, 1)

        # reshape for shaving-eval
        ih, iw = lq.shape[-2:]
        s = math.sqrt(coord.shape[1] / (ih * iw))
        shape = [lq.shape[0], round(ih * s), round(iw * s), 3]
        pred = pred.view(*shape).permute(0, 3, 1, 2).contiguous()
        if gt is not None:
            gt = gt.view(*shape).permute(0, 3, 1, 2).contiguous()

        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, (
                'evaluation with metrics must have gt images.')
            results = dict(eval_result=self.evaluate(pred, gt))
        else:
            results = dict(lq=lq.cpu(), output=pred.cpu())
            if gt is not None:
                results['gt'] = gt.cpu()

        # save image
        if save_image:
            gt_path = meta[0]['gt_path']
            folder_name = osp.splitext(osp.basename(gt_path))[0]
            if isinstance(iteration, numbers.Number):
                save_path = osp.join(save_path, folder_name,
                                     f'{folder_name}-{iteration + 1:06d}.png')
            elif iteration is None:
                save_path = osp.join(save_path, f'{folder_name}.png')
            else:
                raise ValueError('iteration should be number or None, '
                                 f'but got {type(iteration)}')
            mmcv.imwrite(tensor2img(pred), save_path)

        return results

    def val_step(self, data_batch, **kwargs):
        """Validation step.

        Args:
            data_batch (dict): A batch of data.
            kwargs (dict): Other arguments for ``val_step``.

        Returns:
            dict: Returned output.
        """
        output = self.forward_test(**data_batch, **kwargs)
        return output

    def gen_feat(self, input_tensor):
        """Generate feature.

        Args:
            input_tensor (Tensor): input Tensor, shape (B, 3, H, W).

        Returns:
            None.
        """
        feat = self.head(input_tensor)
        self.feat = self.encoder(feat) + feat

    def query_rgb(self, coord, cell=None):
        """Query RGB value of GT.

        Adapted from 'https://github.com/yinboc/liif.git'
        'liif/models/liif.py'
        Copyright (c) 2020, Yinbo Chen, under BSD 3-Clause License.

        Args:
            coord (Tensor): coord tensor, shape (BHW, 2).

        Returns:
            result (Tensor): (part of) output.
        """
        feat = self.feat

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
            pass  # use default initialization
        else:
            raise TypeError('"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')

    @auto_fp16(apply_to=('lq', ))
    def forward(self, lq, gt=None, test_mode=False, **kwargs):
        """Forward function.

        Args:
            lq (Tensor): Input lq images.
            gt (Tensor): Ground-truth image. Default: None.
            test_mode (bool): Whether in test mode or not. Default: False.
            kwargs (dict): Other arguments.
        """

        if test_mode:
            return self.forward_test(lq, gt, **kwargs)

        raise ValueError(
            'LIIF model does not supprot `forward` function in training.',
            'LIIF should be trained by `train_step`.')

    def batched_predict(self, coord, cell):
        """Batched predict.

        Args:
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
                pred = self.query_rgb(coord[:, ql:qr, :], cell[:, ql:qr, :])
                preds.append(pred)
                ql = qr
            pred = torch.cat(preds, dim=1)
        return pred
