# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn


class SpatialTemporalEnsemble(nn.Module):
    """ Apply spatial and temporal ensemble and compute outputs

    Args:
        is_temporal (bool, optional): Whether to apply ensemble temporally. If
            True, the sequence would be flipped temporally. If the input is an
            image, this argument must be set to False. Default: False.

    Returns:
        Tensor: Model outputs given the ensembles.
    """

    def __init__(self, is_temporal_ensemble=False):

        super().__init__()

        self.is_temporal_ensemble = is_temporal_ensemble

    def _transform(self, imgs, mode):
        is_single_image = False
        if imgs.ndim == 4:
            if self.is_temporal_ensemble:
                raise ValueError('"is_temporal_ensemble" must be False if '
                                 'the input is an image.')
            is_single_image = True
            imgs = imgs.unsqueeze(1)

        if mode == 'vertical':
            imgs = imgs[:, :, :, :, ::-1].clone()
        elif mode == 'horizontal':
            imgs = imgs[:, :, :, ::-1, :].clone()
        elif mode == 'transpose':
            imgs = imgs.transpose((0, 1, 2, 4, 3)).clone()

        if is_single_image:
            imgs = imgs.squeeze(1)

        return imgs

    def spatial_ensemble(self, imgs, model):
        img_list = [imgs.cpu()]
        for mode in self.ensemble_mode:
            img_list.extend([self._transform(t, mode) for t in img_list])

        output_list = [model(t.cuda()).cpu() for t in img_list]
        for i in range(len(output_list)):
            if i > 3:
                output_list[i] = self._transform(output_list[i], 'temporal')
            if i % 4 > 1:
                output_list[i] = self._transform(output_list[i], 'horizontal')
            if (i % 4) % 2 == 1:
                output_list[i] = self._transform(output_list[i], 'vertical')

        outputs = torch.stack(output_list, dim=0)
        outputs = outputs.mean(dim=0, keepdim=False)

        return outputs

    def forward(self, imgs, model):
        outputs = self.spatial_ensemble(imgs, model)
        if self.is_temporal_ensemble:
            outputs += self.spatial_ensemble(imgs[:, ::-1], model)[:, ::-1]
            outputs *= 0.5

        return outputs
