import math

import cv2
import mmcv
import numpy as np
from scipy.ndimage.filters import convolve
from scipy.special import gamma

from .metric_utils import gauss_gradient


def sad(alpha, trimap, pred_alpha):
    if alpha.ndim != 2 or trimap.ndim != 2 or pred_alpha.ndim != 2:
        raise ValueError(
            'input alpha, trimap and pred_alpha should has two dimensions, '
            f'alpha {alpha.shape}, please check their shape: '
            f'trimap {trimap.shape}, pred_alpha {pred_alpha.shape}')
    assert (pred_alpha[trimap == 0] == 0).all()
    assert (pred_alpha[trimap == 255] == 255).all()
    alpha = alpha.astype(np.float64) / 255
    pred_alpha = pred_alpha.astype(np.float64) / 255
    sad = np.abs(pred_alpha - alpha).sum() / 1000
    return sad


def mse(alpha, trimap, pred_alpha):
    if alpha.ndim != 2 or trimap.ndim != 2 or pred_alpha.ndim != 2:
        raise ValueError(
            'input alpha, trimap and pred_alpha should has two dimensions, '
            f'alpha {alpha.shape}, please check their shape: '
            f'trimap {trimap.shape}, pred_alpha {pred_alpha.shape}')
    assert (pred_alpha[trimap == 0] == 0).all()
    assert (pred_alpha[trimap == 255] == 255).all()
    alpha = alpha.astype(np.float64) / 255
    pred_alpha = pred_alpha.astype(np.float64) / 255
    weight_sum = (trimap == 128).sum()
    if weight_sum != 0:
        mse = ((pred_alpha - alpha)**2).sum() / weight_sum
    else:
        mse = 0
    return mse


def gradient_error(alpha, trimap, pred_alpha, sigma=1.4):
    """Gradient error for evaluating alpha matte prediction.

    Args:
        alpha (ndarray): Ground-truth alpha matte.
        trimap (ndarray): Input trimap with its value in {0, 128, 255}.
        pred_alpha (ndarray): Predicted alpha matte.
        sigma (float): Standard deviation of the gaussian kernel. Default: 1.4.
    """
    if alpha.ndim != 2 or trimap.ndim != 2 or pred_alpha.ndim != 2:
        raise ValueError(
            'input alpha, trimap and pred_alpha should has two dimensions, '
            f'alpha {alpha.shape}, please check their shape: '
            f'trimap {trimap.shape}, pred_alpha {pred_alpha.shape}')
    if not ((pred_alpha[trimap == 0] == 0).all() and
            (pred_alpha[trimap == 255] == 255).all()):
        raise ValueError(
            'pred_alpha should be masked by trimap before evaluation')
    alpha = alpha.astype(np.float64)
    pred_alpha = pred_alpha.astype(np.float64)
    alpha_normed = np.zeros_like(alpha)
    pred_alpha_normed = np.zeros_like(pred_alpha)
    cv2.normalize(alpha, alpha_normed, 1., 0., cv2.NORM_MINMAX)
    cv2.normalize(pred_alpha, pred_alpha_normed, 1., 0., cv2.NORM_MINMAX)

    alpha_grad = gauss_gradient(alpha_normed, sigma).astype(np.float32)
    pred_alpha_grad = gauss_gradient(pred_alpha_normed,
                                     sigma).astype(np.float32)

    grad_loss = ((alpha_grad - pred_alpha_grad)**2 * (trimap == 128)).sum()
    # same as SAD, divide by 1000 to reduce the magnitude of the result
    return grad_loss / 1000


def connectivity(alpha, trimap, pred_alpha, step=0.1):
    """Connectivity error for evaluating alpha matte prediction.

    Args:
        alpha (ndarray): Ground-truth alpha matte with shape (height, width).
            Value range of alpha is [0, 255].
        trimap (ndarray): Input trimap with shape (height, width). Elements
            in trimap are one of {0, 128, 255}.
        pred_alpha (ndarray): Predicted alpha matte with shape (height, width).
            Value range of pred_alpha is [0, 255].
        step (float): Step of threshold when computing intersection between
            `alpha` and `pred_alpha`.
    """
    if alpha.ndim != 2 or trimap.ndim != 2 or pred_alpha.ndim != 2:
        raise ValueError(
            'input alpha, trimap and pred_alpha should has two dimensions, '
            f'alpha {alpha.shape}, please check their shape: '
            f'trimap {trimap.shape}, pred_alpha {pred_alpha.shape}')
    if not ((pred_alpha[trimap == 0] == 0).all() and
            (pred_alpha[trimap == 255] == 255).all()):
        raise ValueError(
            'pred_alpha should be masked by trimap before evaluation')
    alpha = alpha.astype(np.float32) / 255
    pred_alpha = pred_alpha.astype(np.float32) / 255

    height, width = alpha.shape
    thresh_steps = np.arange(0, 1 + step, step)
    round_down_map = -np.ones_like(alpha)
    for i in range(1, len(thresh_steps)):
        alpha_thresh = alpha >= thresh_steps[i]
        pred_alpha_thresh = pred_alpha >= thresh_steps[i]
        intersection = (alpha_thresh & pred_alpha_thresh).astype(np.uint8)

        # connected components
        _, output, stats, _ = cv2.connectedComponentsWithStats(
            intersection, connectivity=4)
        # start from 1 in dim 0 to exclude background
        size = stats[1:, -1]

        # largest connected component of the intersection
        omega = np.zeros_like(alpha)
        if len(size) != 0:
            max_id = np.argmax(size)
            # plus one to include background
            omega[output == max_id + 1] = 1

        mask = (round_down_map == -1) & (omega == 0)
        round_down_map[mask] = thresh_steps[i - 1]
    round_down_map[round_down_map == -1] = 1

    alpha_diff = alpha - round_down_map
    pred_alpha_diff = pred_alpha - round_down_map
    # only calculate difference larger than or equal to 0.15
    alpha_phi = 1 - alpha_diff * (alpha_diff >= 0.15)
    pred_alpha_phi = 1 - pred_alpha_diff * (pred_alpha_diff >= 0.15)

    connectivity_error = np.sum(
        np.abs(alpha_phi - pred_alpha_phi) * (trimap == 128))
    # same as SAD, divide by 1000 to reduce the magnitude of the result
    return connectivity_error / 1000


def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    if len(img.shape) == 2:
        img = img[..., None]
        return img
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img


def psnr(img1, img2, crop_border=0, input_order='HWC'):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the PSNR calculation. Default: 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.

    Returns:
        float: psnr result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1, img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def ssim(img1, img2, crop_border=0, input_order='HWC'):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the SSIM calculation. Default: 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.

    Returns:
        float: ssim result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))
    return np.array(ssims).mean()


class L1Evaluation(object):
    """L1 evaluation metric.

    Args:
        data_dict (dict): Must contain keys of 'gt_img' and 'fake_res'. If
            'mask' is given, the results will be computed with mask as weight.
    """

    def __call__(self, data_dict):
        gt = data_dict['gt_img']
        pred = data_dict['fake_res']
        mask = data_dict.get('mask', None)

        from mmedit.models.losses.pixelwise_loss import l1_loss
        l1_error = l1_loss(pred, gt, weight=mask, reduction='mean')

        return l1_error


def estimate_aggd_param(block):
    """Estimate AGGD (Asymmetric Generalized Gaussian Distribution) paramters.

    Args:
        block (ndarray): 2D Image block.

    Returns:
        tuple: alpha (float), beta_l (float) and beta_r (float) for the AGGD
            distribution (Estimating the parames in Equation 7 in the paper).
    """
    block = block.flatten()
    gam = np.arange(0.2, 10.001, 0.001)  # len = 9801
    gam_reciprocal = np.reciprocal(gam)
    r_gam = np.square(gamma(gam_reciprocal * 2)) / (
        gamma(gam_reciprocal) * gamma(gam_reciprocal * 3))

    left_std = np.sqrt(np.mean(block[block < 0]**2))
    right_std = np.sqrt(np.mean(block[block > 0]**2))
    gammahat = left_std / right_std
    rhat = (np.mean(np.abs(block)))**2 / np.mean(block**2)
    rhatnorm = (rhat * (gammahat**3 + 1) *
                (gammahat + 1)) / ((gammahat**2 + 1)**2)
    array_position = np.argmin((r_gam - rhatnorm)**2)

    alpha = gam[array_position]
    beta_l = left_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    beta_r = right_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    return (alpha, beta_l, beta_r)


def compute_feature(block):
    """Compute features.

    Args:
        block (ndarray): 2D Image block.

    Returns:
        list: Features with length of 18.
    """
    feat = []
    alpha, beta_l, beta_r = estimate_aggd_param(block)
    feat.extend([alpha, (beta_l + beta_r) / 2])

    # distortions disturb the fairly regular structure of natural images.
    # This deviation can be captured by analyzing the sample distribution of
    # the products of pairs of adjacent coefficients computed along
    # horizontal, vertical and diagonal orientations.
    shifts = [[0, 1], [1, 0], [1, 1], [1, -1]]
    for i in range(len(shifts)):
        shifted_block = np.roll(block, shifts[i], axis=(0, 1))
        alpha, beta_l, beta_r = estimate_aggd_param(block * shifted_block)
        mean = (beta_r - beta_l) * (gamma(2 / alpha) / gamma(1 / alpha))
        feat.extend([alpha, mean, beta_l, beta_r])
    return feat


def niqe_core(img,
              mu_pris_param,
              cov_pris_param,
              gaussian_window,
              block_size_h=96,
              block_size_w=96):
    """Calculate NIQE (Natural Image Quality Evaluator) metric.

    Ref: Making a "Completely Blind" Image Quality Analyzer.
    This implementation could produce almost the same results as the official
    MATLAB codes: http://live.ece.utexas.edu/research/quality/niqe_release.zip

    Note that we do not include block overlap height and width, since they are
    always 0 in the official implementation.

    For good performance, it is advisable by the official implemtation to
    divide the distorted image in to the same size patched as used for the
    construction of multivariate Gaussian model.

    Args:
        img (ndarray): Input image whose quality needs to be computed. The
            image must be a gray or Y (of YCbCr) image with shape (h, w).
            Range [0, 255] with float type.
        mu_pris_param (ndarray): Mean of a pre-defined multivariate Gaussian
            model calculated on the pristine dataset.
        cov_pris_param (ndarray): Covariance of a pre-defined multivariate
            Gaussian model calculated on the pristine dataset.
        gaussian_window (ndarray): A 7x7 Gaussian window used for smoothing the
            image.
        block_size_h (int): Height of the blocks in to which image is divided.
            Default: 96 (the official recommended value).
        block_size_w (int): Width of the blocks in to which image is divided.
            Default: 96 (the official recommended value).
    """
    # crop image
    h, w = img.shape
    num_block_h = math.floor(h / block_size_h)
    num_block_w = math.floor(w / block_size_w)
    img = img[0:num_block_h * block_size_h, 0:num_block_w * block_size_w]

    distparam = []  # dist param is actually the multiscale features.
    for scale in (1, 2):  # perform on two scales (1, 2)
        mu = convolve(img, gaussian_window, mode='nearest')
        sigma = np.sqrt(
            np.abs(
                convolve(np.square(img), gaussian_window, mode='nearest') -
                np.square(mu)))
        # normalize, as in Eq. 1 in the paper
        img_nomalized = (img - mu) / (sigma + 1)

        feat = []
        for idx_w in range(num_block_w):
            for idx_h in range(num_block_h):
                # process ecah block
                block = img_nomalized[idx_h * block_size_h //
                                      scale:(idx_h + 1) * block_size_h //
                                      scale, idx_w * block_size_w //
                                      scale:(idx_w + 1) * block_size_w //
                                      scale]
                feat.append(compute_feature(block))

        distparam.append(np.array(feat))
        # matlab bicubic downsample with anti-aliasing
        # for simplicity, now we use opencv instead, which will result in
        # a slight difference.
        if scale == 1:
            h, w = img.shape
            img = cv2.resize(
                img / 255., (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
            img = img * 255.

    distparam = np.concatenate(distparam, axis=1)

    # fit a MVG (multivariate Gaussian) model to distorted patch features
    mu_distparam = np.nanmean(distparam, axis=0)
    cov_distparam = np.cov(distparam, rowvar=False)  # TODO: use nancov

    # compute niqe quality, Eq. 10 in the paper
    invcov_param = np.linalg.pinv((cov_pris_param + cov_distparam) / 2)
    quality = np.matmul(
        np.matmul((mu_pris_param - mu_distparam), invcov_param),
        np.transpose((mu_pris_param - mu_distparam)))
    quality = np.sqrt(quality)

    return quality


def niqe(img, crop_border, input_order='HWC', convert_to='y'):
    """Calculate NIQE (Natural Image Quality Evaluator) metric.

    Ref: Making a "Completely Blind" Image Quality Analyzer.
    This implementation could produce almost the same results as the official
    MATLAB codes: http://live.ece.utexas.edu/research/quality/niqe_release.zip

    We use the official params estimated from the pristine dataset.
    We use the recommended block size (96, 96) without overlaps.

    Args:
        img (ndarray): Input image whose quality needs to be computed.
            The input image must be in range [0, 255] with float/int type.
            The input_order of image can be 'HW' or 'HWC' or 'CHW'. (BGR order)
            If the input order is 'HWC' or 'CHW', it will be converted to gray
            or Y (of YCbCr) image according to the ``convert_to`` argument.
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the metric calculation.
        input_order (str): Whether the input order is 'HW', 'HWC' or 'CHW'.
            Default: 'HWC'.
        convert_to (str): Whether coverted to 'y' (of MATLAB YCbCr) or 'gray'.
            Default: 'y'.

    Returns:
        float: NIQE result.
    """

    # we use the official params estimated from the pristine dataset.
    niqe_pris_params = np.load('mmedit/core/evaluation/niqe_pris_params.npz')
    mu_pris_param = niqe_pris_params['mu_pris_param']
    cov_pris_param = niqe_pris_params['cov_pris_param']
    gaussian_window = niqe_pris_params['gaussian_window']

    img = img.astype(np.float32)
    if input_order != 'HW':
        img = reorder_image(img, input_order=input_order)
        if convert_to == 'y':
            img = mmcv.bgr2ycbcr(img / 255., y_only=True) * 255.
        elif convert_to == 'gray':
            img = mmcv.bgr2gray(img / 255., cv2.COLOR_BGR2GRAY) * 255.
        img = np.squeeze(img)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border]

    niqe_result = niqe_core(img, mu_pris_param, cov_pris_param,
                            gaussian_window)

    return niqe_result
