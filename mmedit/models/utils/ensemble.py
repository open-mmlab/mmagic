# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn


class SpatialTemporalEnsemble(nn.Module):
    """Apply spatial and temporal ensemble and compute outputs.

    Args:
        is_temporal_ensemble (bool, optional): Whether to apply ensemble
            temporally. If True, the sequence will also be flipped temporally.
            If the input is an image, this argument must be set to False.
            Default: False.
    """

    def __init__(self, is_temporal_ensemble=False):

        super().__init__()

        self.is_temporal_ensemble = is_temporal_ensemble

    def _transform(self, imgs, mode):
        """Apply spatial transform (flip, rotate) to the images.

        Args:
            imgs (torch.Tensor): The images to be transformed/
            mode (str): The mode of transform. Supported values are 'vertical',
                'horizontal', and 'transpose', corresponding to vertical flip,
                horizontal flip, and rotation, respectively.

        Returns:
            torch.Tensor: Output of the model with spatial ensemble applied.
        """

        is_single_image = False
        if imgs.ndim == 4:
            if self.is_temporal_ensemble:
                raise ValueError('"is_temporal_ensemble" must be False if '
                                 'the input is an image.')
            is_single_image = True
            imgs = imgs.unsqueeze(1)

        if mode == 'vertical':
            imgs = imgs.flip(4).clone()
        elif mode == 'horizontal':
            imgs = imgs.flip(3).clone()
        elif mode == 'transpose':
            imgs = imgs.permute(0, 1, 2, 4, 3).clone()

        if is_single_image:
            imgs = imgs.squeeze(1)

        return imgs

    def spatial_ensemble(self, imgs, model):
        """Apply spatial ensemble.

        Args:
            imgs (torch.Tensor): The images to be processed by the model. Its
                size should be either (n, t, c, h, w) or (n, c, h, w).
            model (nn.Module): The model to process the images.

        Returns:
            torch.Tensor: Output of the model with spatial ensemble applied.
        """

        img_list = [imgs.cpu()]
        for mode in ['vertical', 'horizontal', 'transpose']:
            img_list.extend([self._transform(t, mode) for t in img_list])

        output_list = [model(t.to(imgs.device)).cpu() for t in img_list]
        for i in range(len(output_list)):
            if i > 3:
                output_list[i] = self._transform(output_list[i], 'transpose')
            if i % 4 > 1:
                output_list[i] = self._transform(output_list[i], 'horizontal')
            if (i % 4) % 2 == 1:
                output_list[i] = self._transform(output_list[i], 'vertical')

        outputs = torch.stack(output_list, dim=0)
        outputs = outputs.mean(dim=0, keepdim=False)

        return outputs.to(imgs.device)

    def forward(self, imgs, model):
        """Apply spatial and temporal ensemble.

        Args:
            imgs (torch.Tensor): The images to be processed by the model. Its
                size should be either (n, t, c, h, w) or (n, c, h, w).
            model (nn.Module): The model to process the images.

        Returns:
            torch.Tensor: Output of the model with spatial ensemble applied.
        """
        outputs = self.spatial_ensemble(imgs, model)
        if self.is_temporal_ensemble:
            outputs += self.spatial_ensemble(imgs.flip(1), model).flip(1)
            outputs *= 0.5

        return outputs
