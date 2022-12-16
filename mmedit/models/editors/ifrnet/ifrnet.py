# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.models.base_models import BasicInterpolator
from mmedit.registry import MODELS
from mmedit.structures import EditDataSample, PixelData
from .ifrnet_utils import Charbonnier_Ada, Charbonnier_L1, Geometry, Ternary


@MODELS.register_module()
class IFRNet(BasicInterpolator):
    """Base module of IFRNet for video frame interpolation.

    Paper:
        IFRNet: Intermediate Feature Refine Network
                for Efficient Frame Interpolation

    Ref repo: https://github.com/ltkong218/IFRNet

    Args:
        generator (dict): Config for the generator structure.
        pixel_loss (dict): Config for pixel-wise loss.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        interpolation_scale (int): the scale of FPS(out)/FPS(int). Default: 2
        required_frames (int): Required frames in each process. Default: 2
        step_frames (int): Step size of video frame interpolation. Default: 1
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
    """

    def __init__(self,
                 generator,
                 pixel_loss,
                 train_cfg=None,
                 test_cfg=None,
                 interpolation_scale=2,
                 required_frames=2,
                 step_frames=1,
                 init_cfg=None,
                 data_preprocessor=None):
        super().__init__(
            generator=generator,
            pixel_loss=pixel_loss,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            required_frames=required_frames,
            step_frames=step_frames,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)

        self.interpolation_scale = interpolation_scale
        self.l1_loss = Charbonnier_L1()
        self.tr_loss = Ternary(7)
        self.rb_loss = Charbonnier_Ada()
        self.gc_loss = Geometry(3)

    def set_time_embedding(self, inputs):
        """Get time embedding for interpolated frames.

        According to interpolation scale (e.g. 2, 8), time embedding may be
        [1/2] or [1/8, 2/8, 3/8, 4/8, 5/8, 6/8, 7/8]
        """
        times = torch.arange(
            1, self.interpolation_scale) / self.interpolation_scale
        time_embed = times.reshape((self.interpolation_scale - 1, 1, 1, 1))
        time_embed = time_embed.repeat(inputs.shape[0], 1, 1,
                                       1).to(inputs.device)
        return time_embed

    def set_scaled_inputs(self, inputs):
        """Expand input for multiple frame interpolation.

        Args:
            inputs (torch.Tensor): batch input tensor (B, 2, C, H, W)

        Returns:
            img0, img1 (torch.Tensor): batch input tensor (B, T, C, H, W)
        """
        img0 = inputs[:, 0].unsqueeze(dim=1)
        img1 = inputs[:, 1].unsqueeze(dim=1)
        if self.interpolation_scale > 2:
            img0 = img0.repeat(1, self.interpolation_scale - 1, 1, 1, 1)
            img1 = img1.repeat(1, self.interpolation_scale - 1, 1, 1, 1)
        return img0, img1

    def forward_tensor(self, inputs, data_samples=None, **kwargs):
        """Forward tensor. Returns result of simple forward.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.

        Returns:
            Tensor: result of simple forward.
        """
        time_embed = self.set_time_embedding(inputs)
        img0, img1 = self.set_scaled_inputs(inputs)
        B, T, C, H, W = img0.shape
        img0 = img0.view(B * T, C, H, W)
        img1 = img1.view(B * T, C, H, W)
        out_dict = self.generator(img0, img1, time_embed)
        pred_imgs = out_dict['pred_img']
        pred_imgs = pred_imgs.view(B, T, C, H, W)
        return pred_imgs

    def forward_inference(self, inputs, data_samples=None, **kwargs):
        """Forward inference. Returns predictions of validation, testing, and
        simple inference.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.

        Returns:
            List[EditDataSample]: predictions.
        """
        pred_imgs = self.forward_tensor(inputs, data_samples, **kwargs)
        pred_imgs = self.data_preprocessor.destructor(pred_imgs)
        predictions = []
        for idx in range(pred_imgs.shape[0]):
            predictions.append(
                EditDataSample(
                    pred_img=PixelData(
                        data=pred_imgs[idx].squeeze().to('cpu')),
                    metainfo=data_samples[idx].metainfo))
        return predictions

    def forward_train(self, inputs, data_samples=None, **kwargs):
        """Forward training. Returns dict of losses of training.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.

        Returns:
            dict: Dict of losses.
        """
        time_embed = self.set_time_embedding()
        img0, img1 = self.set_scaled_inputs(inputs)
        out_dict = self.generator(img0, img1, time_embed)
        pred_imgs = out_dict['pred_img']
        feats = out_dict['feats']
        # flows0 = out_dict['flows0']
        # flows1 = out_dict['flows1']

        gt_imgs = [data_sample.gt_img.data for data_sample in data_samples]
        batch_gt_data = torch.stack(gt_imgs)
        ft_1, ft_2, ft_3, ft_4 = self.generator.encoder(batch_gt_data)

        loss_rec = self.l1_loss(pred_imgs - batch_gt_data) + \
            self.tr_loss(pred_imgs, batch_gt_data)
        loss_geo = 0.01 * (
            self.gc_loss(feats[0], ft_1) + self.gc_loss(feats[1], ft_2) +
            self.gc_loss(feats[2], ft_3))

        return dict(
            loss=loss_rec + loss_geo, loss_rec=loss_rec, loss_geo=loss_geo)
