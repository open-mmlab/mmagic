# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import cv2
import numpy as np
import scipy.io as sio
import torch
from skimage import img_as_ubyte
from tqdm import tqdm


def export_images(args):
    """Export images from mat file."""

    noisy_dir = osp.join(args.out_dir, 'noisy')
    os.makedirs(noisy_dir, exist_ok=True)
    gt_dir = osp.join(args.out_dir, 'gt')
    os.makedirs(gt_dir, exist_ok=True)

    noisy_matfile = osp.join(args.data_root, 'ValidationNoisyBlocksSrgb.mat')
    gt_matfile = osp.join(args.data_root, 'ValidationGtBlocksSrgb.mat')
    noisy_images = sio.loadmat(noisy_matfile)
    gt_images = sio.loadmat(gt_matfile)

    noisy_images = np.float32(
        np.array(noisy_images['ValidationNoisyBlocksSrgb']))
    noisy_images /= 255.
    gt_images = np.float32(np.array(gt_images['ValidationGtBlocksSrgb']))
    gt_images /= 255.

    # processing noisy images
    print('Exporting', noisy_matfile, 'to', noisy_dir)
    for i in tqdm(range(40)):
        for k in range(32):
            noisy_patch = torch.from_numpy(
                noisy_images[i, k, :, :, :]).unsqueeze(0).permute(0, 3, 1, 2)
            noisy_patch = torch.clamp(noisy_patch, 0,
                                      1).detach().permute(0, 2, 3,
                                                          1).squeeze(0)
            save_path = osp.join(noisy_dir, f'val_{str(i*32+k)}_NOISY.png')
            cv2.imwrite(
                save_path,
                cv2.cvtColor(img_as_ubyte(noisy_patch), cv2.COLOR_RGB2BGR))

    # processing gt images
    print('Exporting', gt_matfile, 'to', gt_dir)
    for i in tqdm(range(40)):
        for k in range(32):
            gt_patch = torch.from_numpy(
                gt_images[i, k, :, :, :]).unsqueeze(0).permute(0, 3, 1, 2)
            gt_patch = torch.clamp(gt_patch, 0,
                                   1).detach().permute(0, 2, 3, 1).squeeze(0)
            save_path = osp.join(gt_dir, f'val_{str(i*32+k)}_GT.png')
            cv2.imwrite(
                save_path,
                cv2.cvtColor(img_as_ubyte(gt_patch), cv2.COLOR_RGB2BGR))

    print('\nFinish exporting images.')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare SIDD test dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-root', help='dataset root')
    parser.add_argument('--out-dir', help='output directory of dataset')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # export images
    export_images(args)
