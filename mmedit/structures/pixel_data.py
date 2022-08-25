# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import mmengine
import numpy as np
import torch


class PixelData(mmengine.structures.PixelData):
    """Data structure for pixel-level annnotations or predictions.

    Different from parent class:
        Support value.ndim == 4 for frames.

    All data items in ``data_fields`` of ``PixelData`` meet the following
    requirements:

    - They all have 3 dimensions in orders of channel, height, and width.
    - They should have the same height and width.

    Examples:
        >>> metainfo = dict(
        ...     img_id=random.randint(0, 100),
        ...     img_shape=(random.randint(400, 600), random.randint(400, 600)))
        >>> image = np.random.randint(0, 255, (4, 20, 40))
        >>> featmap = torch.randint(0, 255, (10, 20, 40))
        >>> pixel_data = PixelData(metainfo=metainfo,
        ...                        image=image,
        ...                        featmap=featmap)
        >>> print(pixel_data)
        >>> (20, 40)

        >>> # slice
        >>> slice_data = pixel_data[10:20, 20:40]
        >>> assert slice_data.shape == (10, 10)
        >>> slice_data = pixel_data[10, 20]
        >>> assert slice_data.shape == (1, 1)
    """

    def __setattr__(self, name: str, value: Union[torch.Tensor, np.ndarray]):
        """Set attributes of ``PixelData``.

        If the dimension of value is 2 and its shape meet the demand, it
        will automatically expend its channel-dimension.

        Args:
            name (str): The key to access the value, stored in `PixelData`.
            value (Union[torch.Tensor, np.ndarray]): The value to store in.
                The type of value must be  `torch.Tensor` or `np.ndarray`,
                and its shape must meet the requirements of `PixelData`.
        """

        if name in ('_metainfo_fields', '_data_fields'):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(
                    f'{name} has been used as a '
                    f'private attribute, which is immutable. ')

        else:
            assert isinstance(value, (torch.Tensor, np.ndarray)), \
                f'Can set {type(value)}, only support' \
                f' {(torch.Tensor, np.ndarray)}'

            if self.shape:
                assert tuple(value.shape[-2:]) == self.shape, (
                    f'the height and width of '
                    f'values {tuple(value.shape[-2:])} is '
                    f'not consistent with'
                    f' the length of this '
                    f':obj:`PixelData` '
                    f'{self.shape} ')
            assert value.ndim in [
                2, 3, 4
            ], f'The dim of value must be 2, 3 or 4, but got {value.ndim}'

            # call BaseDataElement.__setattr__
            super(mmengine.structures.PixelData, self).__setattr__(name, value)
