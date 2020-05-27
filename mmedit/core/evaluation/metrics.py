import cv2
import numpy as np

from .metric_utils import gauss_gradient


def sad(alpha, trimap, pred_alpha):
    assert (pred_alpha[trimap == 0] == 0).all()
    assert (pred_alpha[trimap == 255] == 255).all()
    alpha = alpha.astype(np.float64) / 255
    pred_alpha = pred_alpha.astype(np.float64) / 255
    sad = np.abs(pred_alpha - alpha).sum() / 1000
    return sad


def mse(alpha, trimap, pred_alpha):
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
        img1, img2 (ndarray): Images with range [0, 255].
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
        img1, img2 (ndarray): Images with range [0, 255].
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
