# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import torch

from mmedit.utils import reorder_image


def average(results, key):
    """Average of key in results(list[dict]).

    Args:
        results (list[dict]): A list of dict containing the necessary data.
        key (str): The key of target data.

    Returns:
        result: The average result.
    """

    total = 0
    n = 0
    for batch_result in results:
        batch_size = batch_result.get('batch_size', 1)
        total += batch_result[key] * batch_size
        n += batch_size

    return total / n


def img_transform(img,
                  crop_border=0,
                  input_order='HWC',
                  convert_to=None,
                  channel_order='rgb'):
    """Image transform.

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img (np.ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the PSNR calculation. Default: 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        convert_to (str): Whether to convert the images to other color models.
            If None, the images are not altered. When computing for 'Y',
            the images are assumed to be in BGR order. Options are 'Y' and
            None. Default: None.
        channel_order (str): The channel order of image. Default: 'rgb'

    Returns:
        float: PSNR result.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')

    img = reorder_image(img, input_order=input_order)
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    img = img.astype(np.float32)

    if isinstance(convert_to, str) and convert_to.lower() == 'y':
        if channel_order == 'rgb':
            img = mmcv.rgb2ycbcr(img / 255., y_only=True) * 255.
        elif channel_order == 'bgr':
            img = mmcv.bgr2ycbcr(img / 255., y_only=True) * 255.
        else:
            raise ValueError(
                'Only support `rgb2y` and `bgr2`, but the channel_order '
                f'is {channel_order}')
        img = np.expand_dims(img, axis=2)
    elif convert_to is not None:
        raise ValueError('Wrong color model. Supported values are '
                         '"Y" and None.')

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, None]

    return img


def obtain_data(data_sample, key, device='cpu'):
    """Obtain data of key from data_sample and converse data to device.
    Args:
        data_sample (dict): A dict of data sample.
        key (str): The key of data to obtain.
        device (str): Which device the data will deploy. Default: 'cpu'.

    Returns:
        result (Tensor | np.ndarray): The data of key.
    """
    candidates = ['data_samples', key, 'data']

    for k in candidates:
        if k in data_sample:
            result = data_sample[k]
            if isinstance(result, dict):
                return obtain_data(result, key, device)
            else:
                if isinstance(result, torch.Tensor):
                    result = result.to(device)
                return result

    raise KeyError('Mapping key was not found')
