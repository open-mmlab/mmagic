# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import Dict

from mmcv.transforms import BaseTransform

from mmagic.registry import TRANSFORMS


@TRANSFORMS.register_module()
class CopyValues(BaseTransform):
    """Copy the value of source keys to destination keys.

    # TODO Change to dict(dst=src)

    It does the following: results[dst_key] = results[src_key] for
    (src_key, dst_key) in zip(src_keys, dst_keys).

    Added keys are the keys in the attribute "dst_keys".

    Required Keys:

    - [SRC_KEYS]

    Added Keys:

    - [DST_KEYS]

    Args:
        src_keys (list[str]): The source keys.
        dst_keys (list[str]): The destination keys.
    """

    def __init__(self, src_keys, dst_keys):

        if not isinstance(src_keys, list) or not isinstance(dst_keys, list):
            raise AssertionError('"src_keys" and "dst_keys" must be lists.')

        if len(src_keys) != len(dst_keys):
            raise ValueError('"src_keys" and "dst_keys" should have the same'
                             'number of elements.')

        self.src_keys = src_keys
        self.dst_keys = dst_keys

    def transform(self, results):
        """transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict with a key added/modified.
        """

        for (src_key, dst_key) in zip(self.src_keys, self.dst_keys):
            results[dst_key] = deepcopy(results[src_key])

        return results

    def __repr__(self):

        repr_str = self.__class__.__name__
        repr_str += (f'(src_keys={self.src_keys})')
        repr_str += (f'(dst_keys={self.dst_keys})')

        return repr_str


@TRANSFORMS.register_module()
class SetValues(BaseTransform):
    """Set value to destination keys.

    It does the following: results[key] = value

    Added keys are the keys in the dictionary.

    Required Keys:

    - None

    Added or Modified Keys:

    - keys in the dictionary

    Args:
        dictionary (dict): The dictionary to update.
    """

    def __init__(self, dictionary):

        self.dictionary = dictionary

    def transform(self, results: Dict):
        """transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict with a key added/modified.
        """

        dictionary = deepcopy(self.dictionary)
        results.update(dictionary)

        return results

    def __repr__(self):

        repr_str = self.__class__.__name__
        repr_str += (f'(dictionary={self.dictionary})')

        return repr_str
