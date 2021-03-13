import math
import random

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ..registry import PIPELINES


@PIPELINES.register_module()
class DownSampling:
    """Generate LQ image from GT (and crop).

    Args:
        scale_min (int): The minimum of upsampling scale. Default: 1.
        scale_max (int): The maximum of upsampling scale. Default: 4.
        inp_size (int): The input size, i.e. cropped lr patch size.
            Default: None, means no crop.
    """

    def __init__(self, scale_min=1, scale_max=4, inp_size=None):
        assert scale_max >= scale_min
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.inp_size = inp_size

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        img = results['gt']
        scale = random.uniform(self.scale_min, self.scale_max)
        if self.inp_size is None:
            h_lr = math.floor(img.shape[-3] / scale + 1e-9)
            w_lr = math.floor(img.shape[-2] / scale + 1e-9)
            img = img[:round(h_lr * scale), :round(w_lr * scale), :]
            img_down = resize_fn(img, (w_lr, h_lr))
            crop_lr, crop_hr = img_down, img
        else:
            w_lr = self.inp_size
            w_hr = round(w_lr * scale)
            x0 = random.randint(0, img.shape[-3] - w_hr)
            y0 = random.randint(0, img.shape[-2] - w_hr)
            crop_hr = img[x0:x0 + w_hr, y0:y0 + w_hr, :]
            crop_lr = resize_fn(crop_hr, w_lr)
        results['gt'] = crop_hr
        results['lq'] = crop_lr
        results['scale'] = scale

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'scale_min={self.scale_min}, '
                     f'scale_max={self.scale_max}, '
                     f'inp_size={self.inp_size}')

        return repr_str


def resize_fn(img, size):
    if isinstance(size, int):
        size = (size, size)
    if isinstance(img, np.ndarray):
        return np.asarray(Image.fromarray(img).resize(size, Image.BICUBIC))
    elif isinstance(img, torch.Tensor):
        return transforms.ToTensor()(
            transforms.Resize(size,
                              Image.BICUBIC)(transforms.ToPILImage()(img)))

    else:
        raise TypeError('img should got np.ndarray or torch.Tensor,'
                        f'but got {type(img)}')
