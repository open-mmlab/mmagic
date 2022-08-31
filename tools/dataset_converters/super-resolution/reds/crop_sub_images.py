# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import re
import sys
from multiprocessing import Pool

import cv2
import mmcv
import numpy as np


def worker(path, opt):
    """Worker for each process.

    Args:
        path (str): Image path.
        opt (dict): Configuration dict. It contains:
            crop_size (int): Crop size.
            step (int): Step for overlapped sliding window.
            thresh_size (int): Threshold size. Patches whose size is smaller
                than thresh_size will be dropped.
            save_folder (str): Path to save folder.
            compression_level (int): for cv2.IMWRITE_PNG_COMPRESSION.

    Returns:
        process_info (str): Process information displayed in progress bar.
    """
    crop_size = opt['crop_size']
    step = opt['step']
    thresh_size = opt['thresh_size']
    sequence, img_name = re.split(r'[\\/]', path)[-2:]
    img_name, extension = osp.splitext(osp.basename(path))

    img = mmcv.imread(path, flag='unchanged')

    if img.ndim == 2 or img.ndim == 3:
        h, w = img.shape[:2]
    else:
        raise ValueError(f'Image ndim should be 2 or 3, but got {img.ndim}')

    h_space = np.arange(0, h - crop_size + 1, step)
    if h - (h_space[-1] + crop_size) > thresh_size:
        h_space = np.append(h_space, h - crop_size)
    w_space = np.arange(0, w - crop_size + 1, step)
    if w - (w_space[-1] + crop_size) > thresh_size:
        w_space = np.append(w_space, w - crop_size)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            cropped_img = img[x:x + crop_size, y:y + crop_size, ...]
            sub_folder = osp.join(opt['save_folder'],
                                  f'{sequence}_s{index:03d}')
            mmcv.mkdir_or_exist(sub_folder)
            cv2.imwrite(
                osp.join(sub_folder, f'{img_name}{extension}'), cropped_img,
                [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
    process_info = f'Processing {img_name} ...'
    return process_info


def extract_subimages(opt):
    """Crop images to subimages.

    Args:
        opt (dict): Configuration dict. It contains:
            input_folder (str): Path to the input folder.
            save_folder (str): Path to save folder.
            n_thread (int): Thread number.
    """
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print(f'mkdir {save_folder} ...')
    else:
        print(f'Folder {save_folder} already exists. Exit.')
        sys.exit(1)

    img_list = list(mmcv.scandir(input_folder, recursive=True))

    img_list = [osp.join(input_folder, v) for v in img_list]
    prog_bar = mmcv.ProgressBar(len(img_list))
    pool = Pool(opt['n_thread'])
    for path in img_list:
        pool.apply_async(
            worker, args=(path, opt), callback=lambda arg: prog_bar.update())
    pool.close()
    pool.join()
    print('All processes done.')


def main_extract_subimages(args):
    """A multi-thread tool to crop large images to sub-images for faster IO.

    It is used for REDS dataset.

    opt (dict): Configuration dict. It contains:
        n_thread (int): Thread number.
        compression_level (int):  CV_IMWRITE_PNG_COMPRESSION from 0 to 9.
            A higher value means a smaller size and longer compression time.
            Use 0 for faster CPU decompression. Default: 3, same in cv2.

        scales (list[int]): The downsampling factors corresponding to the
            LR folders you want to process.
            Default: [].
        input_folder (str): Path to the input folder.
        save_folder (str): Path to save folder.
        crop_size (int): Crop size.
        step (int): Step for overlapped sliding window.
        thresh_size (int): Threshold size. Patches whose size is lower
            than thresh_size will be dropped.

    Usage:
        For each folder, run this script.
        For example, if scales = [4], there are two folders to be processed:
            train_sharp
            train_sharp_bicubic/X4
        After process, each sub_folder should have the same number of
        subimages. You can also specify scales by modifying the argument
        'scales'. Remember to modify opt configurations according to your
        settings.
    """

    opt = {}
    opt['n_thread'] = args.n_thread
    opt['compression_level'] = args.compression_level

    # HR images
    opt['input_folder'] = osp.join(args.data_root, 'train_sharp')
    opt['save_folder'] = osp.join(args.data_root, 'train_sharp_sub')
    opt['crop_size'] = args.crop_size
    opt['step'] = args.step
    opt['thresh_size'] = args.thresh_size
    extract_subimages(opt)

    for scale in args.scales:
        opt['input_folder'] = osp.join(args.data_root,
                                       f'train_sharp_bicubic/X{scale}')
        opt['save_folder'] = osp.join(args.data_root,
                                      f'train_sharp_bicubic/X{scale}_sub')
        opt['crop_size'] = args.crop_size // scale
        opt['step'] = args.step // scale
        opt['thresh_size'] = args.thresh_size // scale
        extract_subimages(opt)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocess REDS datasets',
        epilog='You can first download REDS datasets using the script from:'
        'https://gist.github.com/SeungjunNah/b10d369b92840cb8dd2118dd4f41d643')
    parser.add_argument('--data-root', type=str, help='root path for REDS')
    parser.add_argument(
        '--scales', nargs='*', default=[], help='scale factor list')
    parser.add_argument(
        '--crop-size',
        nargs='?',
        default=480,
        help='cropped size for HR images')
    parser.add_argument(
        '--step', nargs='?', default=240, help='step size for HR images')
    parser.add_argument(
        '--thresh-size',
        nargs='?',
        default=0,
        help='threshold size for HR images')
    parser.add_argument(
        '--compression-level',
        nargs='?',
        default=3,
        help='compression level when save png images')
    parser.add_argument(
        '--n-thread',
        nargs='?',
        default=20,
        help='thread number when using multiprocessing')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # extract subimages
    args.scales = [int(v) for v in args.scales]
    main_extract_subimages(args)
