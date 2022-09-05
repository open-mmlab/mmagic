# Copyright (c) OpenMMLab. All rights reserved.
import sys

import mmcv
import mmengine
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.model import is_model_wrapper
from scipy import signal

from mmedit.models.utils import get_module_device


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


def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (np.ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        np.ndarray: Reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    if len(img.shape) == 2:
        img = img[..., None]
        return img
    if input_order == 'CHW':
        if isinstance(img, np.ndarray):
            img = img.transpose(1, 2, 0)
        elif isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0)
    return img


def to_numpy(img, dtype=np.float64):
    """Convert data into numpy arrays of dtype.

    Args:
        img (Tensor | np.ndarray): Input data.
        dtype (np.dtype): Set the data type of the output. Default: np.float64

    Returns:
        img (np.ndarray): Converted numpy arrays data.
    """
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    elif not isinstance(img, np.ndarray):
        raise TypeError('Only support torch.tensor and np.ndarray, '
                        f'but got type {type(img)}')

    img = img.astype(dtype)

    return img


@torch.no_grad()
def extract_inception_features(dataloader,
                               inception,
                               num_samples,
                               inception_style='pytorch'):
    """Extract inception features for FID metric.

    Args:
        dataloader (:obj:`DataLoader`): Dataloader for images.
        inception (nn.Module): Inception network.
        num_samples (int): The number of samples to be extracted.
        inception_style (str): The style of Inception network, "pytorch" or
            "stylegan". Defaults to "pytorch".

    Returns:
        torch.Tensor: Inception features.
    """
    batch_size = dataloader.batch_size
    num_iters = num_samples // batch_size
    if num_iters * batch_size < num_samples:
        num_iters += 1
    # define mmengine progress bar
    pbar = mmengine.ProgressBar(num_iters)

    feature_list = []
    curr_iter = 1
    for data in dataloader:
        # a dirty walkround to support multiple datasets (mainly for the
        # unconditional dataset and conditional dataset). In our
        # implementation, unconditioanl dataset will return real images with
        # the key "real_img". However, the conditional dataset contains a key
        # "img" denoting the real images.
        if 'real_img' in data:
            # Mainly for the unconditional dataset in our MMGeneration
            img = data['real_img']
        else:
            # Mainly for conditional dataset in MMClassification
            img = data['img']
        pbar.update()

        # the inception network is not wrapped with module wrapper.
        if not is_model_wrapper(inception):
            # put the img to the module device
            img = img.to(get_module_device(inception))

        if inception_style == 'stylegan':
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            feature = inception(img, return_features=True)
        else:
            feature = inception(img)[0].view(img.shape[0], -1)
        feature_list.append(feature.to('cpu'))

        if curr_iter >= num_iters:
            break
        curr_iter += 1

    # Attention: the number of features may be different as you want.
    features = torch.cat(feature_list, 0)

    assert features.shape[0] >= num_samples
    features = features[:num_samples]

    # to change the line after pbar
    sys.stdout.write('\n')
    return features


def _hox_downsample(img):
    r"""Downsample images with factor equal to 0.5.

    Ref: https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/ms_ssim.py  # noqa

    Args:
        img (ndarray): Images with order "NHWC".

    Returns:
        ndarray: Downsampled images with order "NHWC".
    """
    return (img[:, 0::2, 0::2, :] + img[:, 1::2, 0::2, :] +
            img[:, 0::2, 1::2, :] + img[:, 1::2, 1::2, :]) * 0.25


def _f_special_gauss(size, sigma):
    r"""Return a circular symmetric gaussian kernel.

    Ref: https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/ms_ssim.py  # noqa

    Args:
        size (int): Size of Gaussian kernel.
        sigma (float): Standard deviation for Gaussian blur kernel.

    Returns:
        ndarray: Gaussian kernel.
    """
    radius = size // 2
    offset = 0.0
    start, stop = -radius, radius + 1
    if size % 2 == 0:
        offset = 0.5
        stop -= 1
    x, y = np.mgrid[offset + start:stop, offset + start:stop]
    assert len(x) == size
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return g / g.sum()


# Gaussian blur kernel
def get_gaussian_kernel():
    kernel = np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6],
                       [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]],
                      np.float32) / 256.0
    gaussian_k = torch.as_tensor(kernel.reshape(1, 1, 5, 5))
    return gaussian_k


def get_pyramid_layer(image, gaussian_k, direction='down'):
    gaussian_k = gaussian_k.to(image.device)
    if direction == 'up':
        image = F.interpolate(image, scale_factor=2)
    multiband = [
        F.conv2d(
            image[:, i:i + 1, :, :],
            gaussian_k,
            padding=2,
            stride=1 if direction == 'up' else 2) for i in range(3)
    ]
    image = torch.cat(multiband, dim=1)
    return image


def gaussian_pyramid(original, n_pyramids, gaussian_k):
    x = original
    # pyramid down
    pyramids = [original]
    for _ in range(n_pyramids):
        x = get_pyramid_layer(x, gaussian_k)
        pyramids.append(x)
    return pyramids


def laplacian_pyramid(original, n_pyramids, gaussian_k):
    """Calculate Laplacian pyramid.

    Ref: https://github.com/koshian2/swd-pytorch/blob/master/swd.py

    Args:
        original (Tensor): Batch of Images with range [0, 1] and order "NCHW"
        n_pyramids (int): Levels of pyramids minus one.
        gaussian_k (Tensor): Gaussian kernel with shape (1, 1, 5, 5).

    Return:
        list[Tensor]. Laplacian pyramids of original.
    """
    # create gaussian pyramid
    pyramids = gaussian_pyramid(original, n_pyramids, gaussian_k)

    # pyramid up - diff
    laplacian = []
    for i in range(len(pyramids) - 1):
        diff = pyramids[i] - get_pyramid_layer(pyramids[i + 1], gaussian_k,
                                               'up')
        laplacian.append(diff)
    # Add last gaussian pyramid
    laplacian.append(pyramids[len(pyramids) - 1])
    return laplacian


def get_descriptors_for_minibatch(minibatch, nhood_size, nhoods_per_image):
    r"""Get descriptors of one level of pyramids.

    Ref: https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/sliced_wasserstein.py  # noqa

    Args:
        minibatch (Tensor): Pyramids of one level with order "NCHW".
        nhood_size (int): Pixel neighborhood size.
        nhoods_per_image (int): The number of descriptors per image.

    Return:
        Tensor: Descriptors of images from one level batch.
    """
    S = minibatch.shape  # (minibatch, channel, height, width)
    assert len(S) == 4 and S[1] == 3
    N = nhoods_per_image * S[0]
    H = nhood_size // 2
    nhood, chan, x, y = np.ogrid[0:N, 0:3, -H:H + 1, -H:H + 1]
    img = nhood // nhoods_per_image
    x = x + np.random.randint(H, S[3] - H, size=(N, 1, 1, 1))
    y = y + np.random.randint(H, S[2] - H, size=(N, 1, 1, 1))
    idx = ((img * S[1] + chan) * S[2] + y) * S[3] + x
    return minibatch.view(-1)[idx]


def finalize_descriptors(desc):
    r"""Normalize and reshape descriptors.

    Ref: https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/sliced_wasserstein.py  # noqa

    Args:
        desc (list or Tensor): List of descriptors of one level.

    Return:
        Tensor: Descriptors after normalized along channel and flattened.
    """
    if isinstance(desc, list):
        desc = torch.cat(desc, dim=0)
    assert desc.ndim == 4  # (neighborhood, channel, height, width)
    desc -= torch.mean(desc, dim=(0, 2, 3), keepdim=True)
    desc /= torch.std(desc, dim=(0, 2, 3), keepdim=True)
    desc = desc.reshape(desc.shape[0], -1)
    return desc


def compute_pr_distances(row_features,
                         col_features,
                         num_gpus=1,
                         rank=0,
                         col_batch_size=10000):
    r"""Compute distances between real images and fake images.

    This function is used for calculate Precision and Recall metric.
    Refer to:https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/metrics/precision_recall.py  # noqa
    """
    assert 0 <= rank < num_gpus
    num_cols = col_features.shape[0]
    num_batches = ((num_cols - 1) // col_batch_size // num_gpus + 1) * num_gpus
    col_batches = torch.nn.functional.pad(col_features,
                                          [0, 0, 0, -num_cols % num_batches
                                           ]).chunk(num_batches)
    dist_batches = []
    for col_batch in col_batches[rank::num_gpus]:
        dist_batch = torch.cdist(
            row_features.unsqueeze(0), col_batch.unsqueeze(0))[0]
        for src in range(num_gpus):
            dist_broadcast = dist_batch.clone()
            if num_gpus > 1:
                torch.distributed.broadcast(dist_broadcast, src=src)
            dist_batches.append(dist_broadcast.cpu() if rank == 0 else None)
    return torch.cat(dist_batches, dim=1)[:, :num_cols] if rank == 0 else None


def normalize(a):
    """L2 normalization.

    Args:
        a (Tensor): Tensor with shape [N, C].

    Returns:
        Tensor: Tensor after L2 normalization per-instance.
    """
    return a / torch.norm(a, dim=1, keepdim=True)


def slerp(a, b, percent):
    """Spherical linear interpolation between two unnormalized vectors.

    Args:
        a (Tensor): Tensor with shape [N, C].
        b (Tensor): Tensor with shape [N, C].
        percent (float|Tensor): A float or tensor with shape broadcastable to
            the shape of input Tensors.

    Returns:
        Tensor: Spherical linear interpolation result with shape [N, C].
    """
    a = normalize(a)
    b = normalize(b)
    d = (a * b).sum(-1, keepdim=True)
    p = percent * torch.acos(d)
    c = normalize(b - d * a)
    d = a * torch.cos(p) + c * torch.sin(p)

    return normalize(d)


def _ssim_for_multi_scale(img1,
                          img2,
                          max_val=255,
                          filter_size=11,
                          filter_sigma=1.5,
                          k1=0.01,
                          k2=0.03):
    """Calculate SSIM (structural similarity) and contrast sensitivity.

    Ref:
    Image quality assessment: From error visibility to structural similarity.

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    This function attempts to match the functionality of ssim_index_new.m by
    Zhou Wang: http://www.cns.nyu.edu/~lcv/ssim/msssim.zip

    Args:
        img1 (ndarray): Images with range [0, 255] and order "NHWC".
        img2 (ndarray): Images with range [0, 255] and order "NHWC".
        max_val (int): the dynamic range of the images (i.e., the difference
            between the maximum the and minimum allowed values).
            Default to 255.
        filter_size (int): Size of blur kernel to use (will be reduced for
            small images). Default to 11.
        filter_sigma (float): Standard deviation for Gaussian blur kernel (will
            be reduced for small images). Default to 1.5.
        k1 (float): Constant used to maintain stability in the SSIM calculation
            (0.01 in the original paper). Default to 0.01.
        k2 (float): Constant used to maintain stability in the SSIM calculation
            (0.03 in the original paper). Default to 0.03.

    Returns:
        tuple: Pair containing the mean SSIM and contrast sensitivity between
        `img1` and `img2`.
    """
    if img1.shape != img2.shape:
        raise RuntimeError(
            'Input images must have the same shape (%s vs. %s).' %
            (img1.shape, img2.shape))
    if img1.ndim != 4:
        raise RuntimeError('Input images must have four dimensions, not %d' %
                           img1.ndim)

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    _, height, width, _ = img1.shape

    # Filter size can't be larger than height or width of images.
    size = min(filter_size, height, width)

    # Scale down sigma if a smaller filter size is used.
    sigma = size * filter_sigma / filter_size if filter_size else 0

    if filter_size:
        window = np.reshape(_f_special_gauss(size, sigma), (1, size, size, 1))
        mu1 = signal.fftconvolve(img1, window, mode='valid')
        mu2 = signal.fftconvolve(img2, window, mode='valid')
        sigma11 = signal.fftconvolve(img1 * img1, window, mode='valid')
        sigma22 = signal.fftconvolve(img2 * img2, window, mode='valid')
        sigma12 = signal.fftconvolve(img1 * img2, window, mode='valid')
    else:
        # Empty blur kernel so no need to convolve.
        mu1, mu2 = img1, img2
        sigma11 = img1 * img1
        sigma22 = img2 * img2
        sigma12 = img1 * img2

    mu11 = mu1 * mu1
    mu22 = mu2 * mu2
    mu12 = mu1 * mu2
    sigma11 -= mu11
    sigma22 -= mu22
    sigma12 -= mu12

    # Calculate intermediate values used by both ssim and cs_map.
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    v1 = 2.0 * sigma12 + c2
    v2 = sigma11 + sigma22 + c2
    ssim = np.mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)),
                   axis=(1, 2, 3))  # Return for each image individually.
    cs = np.mean(v1 / v2, axis=(1, 2, 3))
    return ssim, cs


def ms_ssim(img1,
            img2,
            max_val=255,
            filter_size=11,
            filter_sigma=1.5,
            k1=0.01,
            k2=0.03,
            weights=None,
            reduce_mean=True) -> np.ndarray:
    """Calculate MS-SSIM (multi-scale structural similarity).

    Ref:
    This function implements Multi-Scale Structural Similarity (MS-SSIM) Image
    Quality Assessment according to Zhou Wang's paper, "Multi-scale structural
    similarity for image quality assessment" (2003).
    Link: https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf

    Author's MATLAB implementation:
    http://www.cns.nyu.edu/~lcv/ssim/msssim.zip

    PGGAN's implementation:
    https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/ms_ssim.py

    Args:
        img1 (ndarray): Images with range [0, 255] and order "NHWC".
        img2 (ndarray): Images with range [0, 255] and order "NHWC".
        max_val (int): the dynamic range of the images (i.e., the difference
            between the maximum the and minimum allowed values).
            Default to 255.
        filter_size (int): Size of blur kernel to use (will be reduced for
            small images). Default to 11.
        filter_sigma (float): Standard deviation for Gaussian blur kernel (will
            be reduced for small images). Default to 1.5.
        k1 (float): Constant used to maintain stability in the SSIM calculation
            (0.01 in the original paper). Default to 0.01.
        k2 (float): Constant used to maintain stability in the SSIM calculation
            (0.03 in the original paper). Default to 0.03.
        weights (list): List of weights for each level; if none, use five
            levels and the weights from the original paper. Default to None.

    Returns:
        np.ndarray: MS-SSIM score between `img1` and `img2`.
    """
    if img1.shape != img2.shape:
        raise RuntimeError(
            'Input images must have the same shape (%s vs. %s).' %
            (img1.shape, img2.shape))
    if img1.ndim != 4:
        raise RuntimeError('Input images must have four dimensions, not %d' %
                           img1.ndim)

    # Note: default weights don't sum to 1.0 but do match the paper / matlab
    # code.
    weights = np.array(
        weights if weights else [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = weights.size
    im1, im2 = [x.astype(np.float32) for x in [img1, img2]]
    mssim = []
    mcs = []
    for _ in range(levels):
        ssim, cs = _ssim_for_multi_scale(
            im1,
            im2,
            max_val=max_val,
            filter_size=filter_size,
            filter_sigma=filter_sigma,
            k1=k1,
            k2=k2)
        mssim.append(ssim)
        mcs.append(cs)
        im1, im2 = [_hox_downsample(x) for x in [im1, im2]]

    # Clip to zero. Otherwise we get NaNs.
    mssim = np.clip(np.asarray(mssim), 0.0, np.inf)
    mcs = np.clip(np.asarray(mcs), 0.0, np.inf)

    results = np.prod(mcs[:-1, :]**weights[:-1, np.newaxis], axis=0) * \
        (mssim[-1, :]**weights[-1])
    if reduce_mean:
        # Average over images only at the end.
        results = np.mean(results)
    return results


def sliced_wasserstein(distribution_a,
                       distribution_b,
                       dir_repeats=4,
                       dirs_per_repeat=128):
    r"""sliced Wasserstein distance of two sets of patches.

    Ref: https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/ms_ssim.py  # noqa

    Args:
        distribution_a (Tensor): Descriptors of first distribution.
        distribution_b (Tensor): Descriptors of second distribution.
        dir_repeats (int): The number of projection times. Default to 4.
        dirs_per_repeat (int): The number of directions per projection.
            Default to 128.

    Returns:
        float: sliced Wasserstein distance.
    """
    if torch.cuda.is_available():
        distribution_b = distribution_b.cuda()
    assert distribution_a.ndim == 2
    assert distribution_a.shape == distribution_b.shape
    assert dir_repeats > 0 and dirs_per_repeat > 0
    distribution_a = distribution_a.to(distribution_b.device)
    results = []
    for _ in range(dir_repeats):
        dirs = torch.randn(distribution_a.shape[1], dirs_per_repeat)
        dirs /= torch.sqrt(torch.sum((dirs**2), dim=0, keepdim=True))
        dirs = dirs.to(distribution_b.device)
        proj_a = torch.matmul(distribution_a, dirs)
        proj_b = torch.matmul(distribution_b, dirs)
        # To save cuda memory, we perform sort in cpu
        proj_a, _ = torch.sort(proj_a.cpu(), dim=0)
        proj_b, _ = torch.sort(proj_b.cpu(), dim=0)
        dists = torch.abs(proj_a - proj_b)
        results.append(torch.mean(dists).item())
    torch.cuda.empty_cache()
    return sum(results) / dir_repeats
