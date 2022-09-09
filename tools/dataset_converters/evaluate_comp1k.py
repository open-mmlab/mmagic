# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import re

import cv2
import mmcv
import numpy as np

from mmedit.evaluation import gauss_gradient
from mmedit.utils import modify_args


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
    sad_result = np.abs(pred_alpha - alpha).sum() / 1000
    return sad_result


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
        mse_result = ((pred_alpha - alpha)**2).sum() / weight_sum
    else:
        mse_result = 0
    return mse_result


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


def evaluate_one(args):
    """Function to evaluate one sample of data.

    Args:
        args (tuple): Information needed to evaluate one sample of data.

    Returns:
        dict: The evaluation results including sad, mse, gradient error and
            connectivity error.
    """
    pred_alpha_path, alpha_path, trimap_path = args
    pred_alpha = mmcv.imread(pred_alpha_path, flag='grayscale')
    alpha = mmcv.imread(alpha_path, flag='grayscale')
    if trimap_path is None:
        trimap = np.ones_like(alpha)
    else:
        trimap = mmcv.imread(trimap_path, flag='grayscale')
    sad_result = sad(alpha, trimap, pred_alpha)
    mse_result = mse(alpha, trimap, pred_alpha)
    grad_result = gradient_error(alpha, trimap, pred_alpha)
    conn_result = connectivity(alpha, trimap, pred_alpha)
    return (sad_result, mse_result, grad_result, conn_result)


def evaluate(pred_root, gt_root, trimap_root, verbose, nproc):
    """Evaluate test results of Adobe composition-1k dataset.

    There are 50 different ground truth foregrounds and alpha mattes pairs,
    each of the foreground will be composited with 20 different backgrounds,
    producing 1000 images for testing. In some repo, the ground truth alpha
    matte will be copied 20 times and named the same as the images. This
    function accept both original alpha matte folder (contains 50 ground
    truth alpha mattes) and copied alpha matte folder (contains 1000 ground
    truth alpha mattes) for `gt_root`.

    Example of copied name:
    ```
    alpha_matte1.png -> alpha_matte1_0.png
                        alpha_matte1_1.png
                        ...
                        alpha_matte1_19.png
                        alpha_matte1_20.png
    ```

    Args:
        pred_root (str): Path to the predicted alpha matte folder.
        gt_root (str): Path to the ground truth alpha matte folder.
        trimap_root (str): Path to the predicted alpha matte folder.
        verbose (bool): Whether print result for each predicted alpha matte.
        nproc (int): number of processers.
    """

    images = sorted(mmcv.scandir(pred_root))
    gt_files_num = len(list(mmcv.scandir(gt_root)))
    # If ground truth alpha mattes are not copied (number of files is 50), we
    # use the below pattern to recover the name of the original alpha matte.
    if gt_files_num == 50:
        pattern = re.compile(r'(.+)_(?:\d+)(.png)')
    pairs = []
    for img in images:
        pred_alpha_path = osp.join(pred_root, img)
        # if ground truth alpha matte are not copied, recover the original name
        if gt_files_num == 50:
            groups = pattern.match(img).groups()
            alpha_path = osp.join(gt_root, ''.join(groups))
        # if ground truth alpha matte are copied, the name should be the same
        else:  # gt_files_num == 1000
            alpha_path = osp.join(gt_root, img)
        trimap_path = (
            osp.join(trimap_root, img) if trimap_root is not None else None)
        pairs.append((pred_alpha_path, alpha_path, trimap_path))

    results = mmcv.track_parallel_progress(evaluate_one, pairs, nproc)

    if verbose:
        # for sad_result, mse_result, grad_result, conn_result in results:
        for i, img in enumerate(images):
            sad_result, mse_result, grad_result, conn_result = results[i]
            print(f'{img} SAD: {sad_result:.6g} MattingMSE: {mse_result:.6g} '
                  f'GRAD: {grad_result:.6g} CONN: {conn_result:.6g}')

    sad_mean, mse_mean, grad_mean, conn_mean = np.mean(results, axis=0)
    print(f'MEAN:  SAD: {sad_mean:.6g} MattingMSE: {mse_mean:.6g} '
          f'GRAD: {grad_mean:.6g} CONN: {conn_mean:.6g}')


def parse_args():
    modify_args()
    parser = argparse.ArgumentParser(
        description='evaluate composition-1k prediction result')
    parser.add_argument(
        'pred_root', help='Path to the predicted alpha matte folder')
    parser.add_argument(
        'gt_root', help='Path to the ground truth alpha matte folder')
    parser.add_argument(
        '--trimap-root',
        help='Path to trimap folder. If not specified, '
        'results are calculated on the full image.')
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Whether print result for each predicted alpha matte')
    parser.add_argument(
        '--nproc', type=int, default=4, help='number of processers')
    return parser.parse_args()


def main():
    args = parse_args()

    if not osp.exists(args.pred_root):
        raise FileNotFoundError(f'pred_root {args.pred_root} not found')
    if not osp.exists(args.gt_root):
        raise FileNotFoundError(f'gt_root {args.gt_root} not found')

    evaluate(args.pred_root, args.gt_root, args.trimap_root, args.verbose,
             args.nproc)


if __name__ == '__main__':
    main()
