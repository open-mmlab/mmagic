# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.transforms import BaseTransform

from mmedit.registry import TRANSFORMS


@TRANSFORMS.register_module()
class MirrorSequence(BaseTransform):
    """Extend short sequences (e.g. Vimeo-90K) by mirroring the sequences.

    Given a sequence with N frames (x1, ..., xN), extend the sequence to
    (x1, ..., xN, xN, ..., x1).

    Required Keys:

    - [KEYS]

    Modified Keys:

    - [KEYS]

    Args:
        keys (list[str]): The frame lists to be extended.
    """

    def __init__(self, keys):

        self.keys = keys

    def transform(self, results):
        """transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        for key in self.keys:
            if isinstance(results[key], list):
                results[key] = results[key] + results[key][::-1]
            else:
                raise TypeError('The input must be of class list[nparray]. '
                                f'Got {type(results[key])}.')

        return results

    def __repr__(self):

        repr_str = self.__class__.__name__
        repr_str += (f'(keys={self.keys})')

        return repr_str


@TRANSFORMS.register_module()
class TemporalReverse(BaseTransform):
    """Reverse frame lists for temporal augmentation.

    Required keys are the keys in attributes "lq" and "gt",
    added or modified keys are "lq", "gt" and "reverse".

    Args:
        keys (list[str]): The frame lists to be reversed.
        reverse_ratio (float): The probability to reverse the frame lists.
            Default: 0.5.
    """

    def __init__(self, keys, reverse_ratio=0.5):

        self.keys = keys
        self.reverse_ratio = reverse_ratio

    def transform(self, results):
        """transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        reverse = np.random.random() < self.reverse_ratio

        if reverse:
            for key in self.keys:
                results[key].reverse()

        results['reverse'] = reverse

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(keys={self.keys}, reverse_ratio={self.reverse_ratio})'
        return repr_str
