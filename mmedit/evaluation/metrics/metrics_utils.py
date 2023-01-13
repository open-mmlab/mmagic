# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import torch

from mmedit.utils import reorder_image


def _assert_ndim(input, name, ndim, shape_hint):
    if input.ndim != ndim:
        raise ValueError(
            f'{name} should be of shape {shape_hint}, but got {input.shape}.')


def _assert_masked(pred_alpha, trimap):
    if (pred_alpha[trimap == 0] != 0).any() or (pred_alpha[trimap == 255] !=
                                                255).any():
        raise ValueError(
            'pred_alpha should be masked by trimap before evaluation')


def _fetch_data_and_check(data_samples):
    """Fetch and check data from one item of data_batch and predictions.

    Args:
        data_batch (dict): One item of data_batch.
        predictions (dict): One item of predictions.

    Returns:
        pred_alpha (Tensor): Pred_alpha data of predictions.
        ori_alpha (Tensor): Ori_alpha data of data_batch.
        ori_trimap (Tensor): Ori_trimap data of data_batch.
    """
    ori_trimap = data_samples['ori_trimap'][:, :, 0]
    ori_alpha = data_samples['ori_alpha'][:, :, 0]
    pred_alpha = data_samples['output']['pred_alpha']['data']  # 2D tensor
    pred_alpha = pred_alpha.cpu().numpy()

    _assert_ndim(ori_trimap, 'trimap', 2, 'HxW')
    _assert_ndim(ori_alpha, 'gt_alpha', 2, 'HxW')
    _assert_ndim(pred_alpha, 'pred_alpha', 2, 'HxW')
    _assert_masked(pred_alpha, ori_trimap)

    # dtype uint8 -> float64
    pred_alpha = pred_alpha / 255.0
    ori_alpha = ori_alpha / 255.0
    # test shows that using float32 vs float64 differs final results at 1e-4
    # speed are comparable, so we choose float64 for accuracy

    return pred_alpha, ori_alpha, ori_trimap


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
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]

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
