# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, List, Tuple

import numpy as np
import torch
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform

from mmedit.registry import TRANSFORMS
from mmedit.structures import EditDataSample, PixelData


def check_if_image(value: Any) -> bool:
    """Check if the  input value is image or images.

    If value is a list or Tuple,
    recursively check if  each element in ``value`` is image.

    Args:
        value (Any): The value to be checked.

    Returns:
        bool: If the value is image or sequence of images.
    """

    if isinstance(value, (List, Tuple)):
        is_image = (len(value) > 0)
        for v in value:
            is_image = is_image and check_if_image(v)

    else:
        is_image = isinstance(value, np.ndarray) and len(value.shape) > 1

    return is_image


def image_to_tensor(img):
    """Trans image to tensor.

    Args:
        img (np.ndarray): The original image.

    Returns:
        Tensor: The output tensor.
    """

    if len(img.shape) < 3:
        img = np.expand_dims(img, -1)
    img = np.ascontiguousarray(img.transpose(2, 0, 1))
    tensor = to_tensor(img)

    return tensor


def images_to_tensor(value):
    """Trans image and sequence of frames to tensor.

    Args:
        value (np.ndarray | list[np.ndarray] | Tuple[np.ndarray]):
            The original image or list of frames.

    Returns:
        Tensor: The output tensor.
    """

    if isinstance(value, (List, Tuple)):
        # sequence of frames
        frames = [image_to_tensor(v) for v in value]
        tensor = torch.stack(frames, dim=0)
    elif isinstance(value, np.ndarray):
        tensor = image_to_tensor(value)
    else:
        # Maybe the data has been converted to Tensor.
        tensor = to_tensor(value)

    return tensor


def can_convert_to_image(value):
    """Judge whether the input value can be converted to image tensor via
    :func:`images_to_tensor` function.

    Args:
        value (any): The input value.

    Returns:
        bool: If true, the input value can convert to image with
            :func:`images_to_tensor`, and vice versa.
    """
    if isinstance(value, (List, Tuple)):
        return all([can_convert_to_image(v) for v in value])
    elif isinstance(value, np.ndarray):
        return True
    elif isinstance(value, torch.Tensor):
        return True
    else:
        return False


@TRANSFORMS.register_module()
class PackEditInputs(BaseTransform):
    """Pack the inputs data for SR, VFI, matting and inpainting.

    Keys for images include ``img``, ``gt``, ``ref``, ``mask``, ``gt_heatmap``,
        ``trimap``, ``gt_alpha``, ``gt_fg``, ``gt_bg``. All of them will be
        packed into data field of EditDataSample.
    pack_all (bool): Whether pack all variables in `results` to `inputs` dict.
        This is useful when keys of the input dict is not fixed.
        Please be careful when using this function, because we do not
        Defaults to False.

    Others will be packed into metainfo field of EditDataSample.
    """

    def __init__(self,
                 keys: Tuple[List[str], str, None] = None,
                 pack_all: bool = False):
        if keys is not None:
            if isinstance(keys, list):
                self.keys = keys
            else:
                self.keys = [keys]
        else:
            self.keys = None
        self.pack_all = pack_all

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_samples' (obj:`EditDataSample`): The annotation info of the
                sample.
        """

        packed_results = dict()
        data_sample = EditDataSample()

        pack_keys = [k for k in results.keys()] if self.pack_all else self.keys
        if pack_keys is not None:
            packed_results['inputs'] = dict()
            for key in pack_keys:
                val = results[key]
                if can_convert_to_image(val):
                    packed_results['inputs'][key] = images_to_tensor(val)
                    results.pop(key)

        elif 'img' in results:
            img = results.pop('img')
            img_tensor = images_to_tensor(img)
            packed_results['inputs'] = img_tensor
            data_sample.input = PixelData(data=img_tensor.clone())

        if 'gt' in results:
            gt = results.pop('gt')
            gt_tensor = images_to_tensor(gt)
            if len(gt_tensor.shape) > 3 and gt_tensor.size(0) == 1:
                gt_tensor.squeeze_(0)
            data_sample.gt_img = PixelData(data=gt_tensor)

        if 'gt_label' in results:
            gt_label = results.pop('gt_label')
            data_sample.set_gt_label(gt_label)

        if 'img_lq' in results:
            img_lq = results.pop('img_lq')
            img_lq_tensor = images_to_tensor(img_lq)
            data_sample.img_lq = PixelData(data=img_lq_tensor)

        if 'ref' in results:
            ref = results.pop('ref')
            ref_tensor = images_to_tensor(ref)
            data_sample.ref_img = PixelData(data=ref_tensor)

        if 'ref_lq' in results:
            ref_lq = results.pop('ref_lq')
            ref_lq_tensor = images_to_tensor(ref_lq)
            data_sample.ref_lq = PixelData(data=ref_lq_tensor)

        if 'mask' in results:
            mask = results.pop('mask')
            mask_tensor = images_to_tensor(mask)
            data_sample.mask = PixelData(data=mask_tensor)

        if 'gt_heatmap' in results:
            gt_heatmap = results.pop('gt_heatmap')
            gt_heatmap_tensor = images_to_tensor(gt_heatmap)
            data_sample.gt_heatmap = PixelData(data=gt_heatmap_tensor)

        if 'gt_unsharp' in results:
            gt_unsharp = results.pop('gt_unsharp')
            gt_unsharp_tensor = images_to_tensor(gt_unsharp)
            data_sample.gt_unsharp = PixelData(data=gt_unsharp_tensor)

        if 'merged' in results:
            # image in matting annotation is named merged
            img = results.pop('merged')
            img_tensor = images_to_tensor(img)
            # used for model inputs
            packed_results['inputs'] = img_tensor
            # used as ground truth for composition losses
            data_sample.gt_merged = PixelData(data=img_tensor.clone())

        if 'trimap' in results:
            trimap = results.pop('trimap')
            trimap_tensor = images_to_tensor(trimap)
            data_sample.trimap = PixelData(data=trimap_tensor)

        if 'alpha' in results:
            # gt_alpha in matting annotation is named alpha
            gt_alpha = results.pop('alpha')
            gt_alpha_tensor = images_to_tensor(gt_alpha)
            data_sample.gt_alpha = PixelData(data=gt_alpha_tensor)

        if 'fg' in results:
            # gt_fg in matting annotation is named fg
            gt_fg = results.pop('fg')
            gt_fg_tensor = images_to_tensor(gt_fg)
            data_sample.gt_fg = PixelData(data=gt_fg_tensor)

        if 'bg' in results:
            # gt_bg in matting annotation is named bg
            gt_bg = results.pop('bg')
            gt_bg_tensor = images_to_tensor(gt_bg)
            data_sample.gt_bg = PixelData(data=gt_bg_tensor)

        if 'rgb_img' in results:
            gt_rgb = results.pop('rgb_img')
            gt_rgb_tensor = images_to_tensor(gt_rgb)
            data_sample.gt_rgb = PixelData(data=gt_rgb_tensor)

        if 'gray_img' in results:
            gray = results.pop('gray_img')
            gray_tensor = images_to_tensor(gray)
            data_sample.gray = PixelData(data=gray_tensor)

        if 'cropped_img' in results:
            cropped_img = results.pop('cropped_img')
            cropped_img = images_to_tensor(cropped_img)
            data_sample.cropped_img = PixelData(data=cropped_img)

        metainfo = dict()
        for key in results:
            metainfo[key] = results[key]

        data_sample.set_metainfo(metainfo=metainfo)

        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:

        repr_str = self.__class__.__name__

        return repr_str


@TRANSFORMS.register_module()
class ToTensor(BaseTransform):
    """Convert some values in results dict to `torch.Tensor` type in data
    loader pipeline.

    Args:
        keys (Sequence[str]): Required keys to be converted.
        to_float32 (bool): Whether convert tensors of images to float32.
            Default: True.
    """

    def __init__(self, keys, to_float32=True):

        self.keys = keys
        self.to_float32 = to_float32

    def _data_to_tensor(self, value):
        """Convert the value to tensor."""
        is_image = check_if_image(value)

        if is_image:
            tensor = images_to_tensor(value)
            if self.to_float32:
                tensor = tensor.float()
            if len(tensor.shape) > 3 and tensor.size(0) == 1:
                tensor.squeeze_(0)

        else:
            tensor = to_tensor(value)

        return tensor

    def transform(self, results):
        """transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        for key in self.keys:
            results[key] = self._data_to_tensor(results[key])

        return results

    def __repr__(self):

        return self.__class__.__name__ + (
            f'(keys={self.keys}, to_float32={self.to_float32})')
