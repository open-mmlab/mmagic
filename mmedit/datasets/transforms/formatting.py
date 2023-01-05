# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, List, Tuple

import numpy as np
import torch
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform

from mmedit.registry import TRANSFORMS
from mmedit.structures import EditDataSample


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
    """Pack data into EditDataSample for training, evaluation and testing.

    MMediting follows the design of data structure from MMEngine.
        Data from the loader will be packed into data field of EditDataSample.
        More details of DataSample refer to the documentation of MMEngine:
        https://mmengine.readthedocs.io/en/latest/advanced_tutorials/data_element.html

    Args:
        keys Tuple[List[str], str, None]: The keys to saved in returned
            inputs, which are used as the input of models, default to
            ['img', 'noise', 'merged'].
        data_keys Tuple[List[str], str, None]: The keys to saved in
            `data_field` of the `data_samples`.
        meta_keys Tuple[List[str], str, None]: The meta keys to saved
            in `metainfo` of the `data_samples`. All the other data will
            be packed into the data of the `data_samples`
    """

    def __init__(
        self,
        keys: Tuple[List[str], str] = ['img', 'noise', 'merged'],
        data_keys: Tuple[List[str], str, None] = None,
        meta_keys: Tuple[List[str], str, None] = None,
    ) -> None:

        assert keys is not None, 'keys in PackEditInputs can not be None.'
        self.keys = keys if isinstance(keys, List) else [keys]
        self.data_keys = data_keys
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict: A dict contains

            - 'inputs' (obj:`dict`): The forward data of models.
              According to different tasks, the `inputs` may contain images,
              videos, labels, text, etc.

            - 'data_samples' (obj:`EditDataSample`): The annotation info of the
                sample.
        """

        # prepare inputs
        inputs = dict()
        for key in self.keys:
            value = results.pop(key, None)
            if value is not None and can_convert_to_image(value):
                inputs[key] = images_to_tensor(value)
                if len(value.shape) > 3 and value.size(0) == 1:
                    value.squeeze_(0)
        if len(inputs.values()) == 1:
            inputs = inputs.values()[0]

        # prepare DataSample
        data_sample = EditDataSample()
        for key in self.data_keys:
            value = results.pop(key, None)
            if value is not None:
                if key == 'gt_label':
                    data_sample.set_gt_label(value)
                    continue
                if can_convert_to_image(value):
                    value = images_to_tensor(value)
                    if len(value.shape) > 3 and value.size(0) == 1:
                        value.squeeze_(0)
                data_sample.set_data({key: value})

        # prepare metainfo
        for key in self.meta_keys:
            value = results.pop(key, None)
            if value is not None:
                data_sample.set_metainfo({key: value})

        # set data_sample to None if it has no items
        if len(data_sample.all_items()) == 0:
            data_sample = None

        return {'inputs': inputs, 'data_samples': data_sample}

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
