# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform

from mmedit.registry import TRANSFORMS
from mmedit.structures import EditDataSample
from mmedit.utils import can_convert_to_image, check_if_image, images_to_tensor


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
        data_keys: Tuple[List[str], str] = [],
        meta_keys: Tuple[List[str], str] = [],
    ) -> None:

        assert keys is not None, \
            'keys in PackEditInputs can not be None.'
        assert data_keys is not None, \
            'data_keys in PackEditInputs can not be None.'
        assert meta_keys is not None, \
            'meta_keys in PackEditInputs can not be None.'

        self.keys = keys if isinstance(keys, List) else [keys]
        self.data_keys = data_keys if isinstance(data_keys,
                                                 List) else [data_keys]
        self.meta_keys = meta_keys if isinstance(meta_keys,
                                                 List) else [meta_keys]

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
                value = images_to_tensor(value)
                if len(value.shape) > 3 and value.size(0) == 1:
                    value.squeeze_(0)
                inputs[key] = value

        # return the inputs as tensor, if it has only one item
        if len(inputs.values()) == 1:
            inputs = list(inputs.items())[0]

        data_sample = EditDataSample()
        # prepare metainfo and data in DataSample according to predefined keys
        predefined_data = {
            k: v
            for (k, v) in results.items()
            if not (k in self.data_keys + self.meta_keys)
        }
        data_sample.set_predefined_data(predefined_data)

        # prepare metainfo in DataSample according to user-provided meta_keys
        required_metainfo = {
            k: v
            for (k, v) in results.items() if k in self.meta_keys
        }
        data_sample.set_metainfo(required_metainfo)

        # prepare metainfo in DataSample according to user-provided data_keys
        required_data = {
            k: v
            for (k, v) in results.items() if k in self.data_keys
        }
        data_sample.set_tensor_data(required_data)

        # set data_sample to None if it has no items
        if len(list(data_sample.all_items())) == 0:
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
