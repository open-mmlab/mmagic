# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Sequence

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from torch.nn import functional as F

from ..registry import PIPELINES


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    if isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    if isinstance(data, int):
        return torch.LongTensor([data])
    if isinstance(data, float):
        return torch.FloatTensor([data])

    raise TypeError(f'type {type(data)} cannot be converted to tensor.')


@PIPELINES.register_module()
class ToTensor:
    """Convert some values in results dict to `torch.Tensor` type in data
    loader pipeline.

    Args:
        keys (Sequence[str]): Required keys to be converted.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        for key in self.keys:
            results[key] = to_tensor(results[key])
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@PIPELINES.register_module()
class ImageToTensor:
    """Convert image type to `torch.Tensor` type.

    Args:
        keys (Sequence[str]): Required keys to be converted.
        to_float32 (bool): Whether convert numpy image array to np.float32
            before converted to tensor. Default: True.
    """

    def __init__(self, keys, to_float32=True):
        self.keys = keys
        self.to_float32 = to_float32

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        for key in self.keys:
            # deal with gray scale img: expand a color channel
            if len(results[key].shape) == 2:
                results[key] = results[key][..., None]
            if self.to_float32 and not isinstance(results[key], np.float32):
                results[key] = results[key].astype(np.float32)
            results[key] = to_tensor(results[key].transpose(2, 0, 1))
        return results

    def __repr__(self):
        return self.__class__.__name__ + (
            f'(keys={self.keys}, to_float32={self.to_float32})')


@PIPELINES.register_module()
class FramesToTensor(ImageToTensor):
    """Convert frames type to `torch.Tensor` type.

    It accepts a list of frames, converts each to `torch.Tensor` type and then
    concatenates in a new dimension (dim=0).

    Args:
        keys (Sequence[str]): Required keys to be converted.
        to_float32 (bool): Whether convert numpy image array to np.float32
            before converted to tensor. Default: True.
    """

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        for key in self.keys:
            if not isinstance(results[key], list):
                raise TypeError(f'results["{key}"] should be a list, '
                                f'but got {type(results[key])}')
            for idx, v in enumerate(results[key]):
                # deal with gray scale img: expand a color channel
                if len(v.shape) == 2:
                    v = v[..., None]
                if self.to_float32 and not isinstance(v, np.float32):
                    v = v.astype(np.float32)
                results[key][idx] = to_tensor(v.transpose(2, 0, 1))
            results[key] = torch.stack(results[key], dim=0)
            if results[key].size(0) == 1:
                results[key].squeeze_()
        return results


@PIPELINES.register_module()
class GetMaskedImage:
    """Get masked image.

    Args:
        img_name (str): Key for clean image.
        mask_name (str): Key for mask image. The mask shape should be
            (h, w, 1) while '1' indicate holes and '0' indicate valid
            regions.
    """

    def __init__(self, img_name='gt_img', mask_name='mask'):
        self.img_name = img_name
        self.mask_name = mask_name

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        clean_img = results[self.img_name]
        mask = results[self.mask_name]

        masked_img = clean_img * (1. - mask)
        results['masked_img'] = masked_img

        return results

    def __repr__(self):
        return self.__class__.__name__ + (
            f"(img_name='{self.img_name}', mask_name='{self.mask_name}')")


@PIPELINES.register_module()
class FormatTrimap:
    """Convert trimap (tensor) to one-hot representation.

    It transforms the trimap label from (0, 128, 255) to (0, 1, 2). If
    ``to_onehot`` is set to True, the trimap will convert to one-hot tensor of
    shape (3, H, W). Required key is "trimap", added or modified key are
    "trimap" and "to_onehot".

    Args:
        to_onehot (bool): whether convert trimap to one-hot tensor. Default:
            ``False``.
    """

    def __init__(self, to_onehot=False):
        self.to_onehot = to_onehot

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        trimap = results['trimap'].squeeze()
        trimap[trimap == 128] = 1
        trimap[trimap == 255] = 2
        if self.to_onehot:
            trimap = F.one_hot(trimap.to(torch.long), num_classes=3)
            trimap = trimap.permute(2, 0, 1)
        else:
            trimap = trimap[None, ...]  # expand the channels dimension
        results['trimap'] = trimap.float()
        results['meta'].data['to_onehot'] = self.to_onehot
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(to_onehot={self.to_onehot})'


@PIPELINES.register_module()
class Collect:
    """Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "gt_labels".

    The "img_meta" item is always populated.  The contents of the "meta"
    dictionary depends on "meta_keys".

    Args:
        keys (Sequence[str]): Required keys to be collected.
        meta_keys (Sequence[str]): Required keys to be collected to "meta".
            Default: None.
    """

    def __init__(self, keys, meta_keys=None):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        data = {}
        img_meta = {}
        for key in self.meta_keys:
            img_meta[key] = results[key]
        data['meta'] = DC(img_meta, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + (
            f'(keys={self.keys}, meta_keys={self.meta_keys})')
