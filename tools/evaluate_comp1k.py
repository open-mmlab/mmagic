# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import re

import mmcv
import numpy as np

from mmedit.core.evaluation import connectivity, gradient_error, mse, sad
from mmedit.utils import modify_args


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
            print(f'{img} SAD: {sad_result:.6g} MSE: {mse_result:.6g} '
                  f'GRAD: {grad_result:.6g} CONN: {conn_result:.6g}')

    sad_mean, mse_mean, grad_mean, conn_mean = np.mean(results, axis=0)
    print(f'MEAN:  SAD: {sad_mean:.6g} MSE: {mse_mean:.6g} '
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
