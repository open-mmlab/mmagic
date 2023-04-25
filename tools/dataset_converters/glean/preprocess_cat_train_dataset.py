# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from multiprocessing import Pool

import lmdb
import mmengine
import numpy as np
from PIL import Image
from skimage import img_as_float
from skimage.io import imread, imsave

from mmagic.datasets.transforms import MATLABLikeResize


def export_images(lmdb_path, meta_file_path, out_dir):
    """Export images from lmdb file.

    Ref: https://github.com/fyu/lsun

    Args:
        lmdb_path (str): Lmdb file path.
        meta_file_path (str): Meta file path.
        out_dir (str): Output directory of dataset.
    """

    print('Exporting', lmdb_path, 'to', out_dir)
    env = lmdb.open(
        lmdb_path, map_size=1099511627776, max_readers=100, readonly=True)
    meta_file = open(meta_file_path, 'r')
    img_list = [_.split('.')[0] for _ in meta_file.readlines()]
    txn = env.begin(write=False)
    cursor = txn.cursor()
    count = 0
    for key, val in cursor:
        if key.decode('ascii') in img_list:
            image_out_path = osp.join(out_dir, key.decode('ascii') + '.webp')
            with open(image_out_path, 'wb') as fp:
                fp.write(val)
        count += 1
        if count % 1000 == 0:
            print('Finished', count, 'images')
        if count > len(img_list):
            break
    print('\nFinish exporting images.')


def imresize(img_path, output_path, scale=None, output_shape=None):
    """Resize the image using MATLAB-like downsampling.

    Args:
        img_path (str): Input image path.
        output_path (str): Output image path.
        scale (float | None, optional): The scale factor of the resize
            operation. If None, it will be determined by output_shape.
            Default: None.
        output_shape (tuple(int) | None, optional): The size of the output
            image. If None, it will be determined by scale. Note that if
            scale is provided, output_shape will not be used.
            Default: None.
    """

    matlab_resize = MATLABLikeResize(
        keys=['data'], scale=scale, output_shape=output_shape)
    img = imread(img_path)
    img = img_as_float(img)
    data = {'data': img}
    output = matlab_resize(data)['data']
    output = np.clip(output, 0.0, 1.0) * 255
    output = np.around(output).astype(np.uint8)
    imsave(output_path, output)


def worker(img_name, out_dir):
    """Worker for each process.

    Args:
        img_name (str): Image filename.
        out_dir (str): Output directory of dataset.

    Returns:
        process_info (str): Process information displayed in progress bar.
    """

    _gt_dir = osp.join(out_dir, '_GT')
    gt_dir = osp.join(out_dir, 'GT')
    bix8_dir = osp.join(out_dir, 'BIx8_down')
    bix16_dir = osp.join(out_dir, 'BIx16_down')

    new_img_name = img_name + '.png'
    img = Image.open(osp.join(_gt_dir, img_name + '.webp'))
    img.load()
    img.save(osp.join(_gt_dir, new_img_name))

    imresize(
        osp.join(_gt_dir, new_img_name),
        osp.join(gt_dir, new_img_name),
        output_shape=(256, 256))
    imresize(
        osp.join(gt_dir, new_img_name),
        osp.join(bix8_dir, new_img_name),
        scale=1 / 8)
    imresize(
        osp.join(gt_dir, new_img_name),
        osp.join(bix16_dir, new_img_name),
        scale=1 / 16)

    process_info = f'Processing {new_img_name} ...'
    return process_info


def downsample_images(args):
    """Downsample images for cat_train datasets from the ground-truth."""

    meta_file = open(args.meta_file_path, 'r')
    img_list = [_.split('.')[0] for _ in meta_file.readlines()]
    prog_bar = mmengine.ProgressBar(len(img_list))
    pool = Pool(args.n_thread)
    for path in img_list:
        pool.apply_async(
            worker,
            args=(path, args.out_dir),
            callback=lambda arg: prog_bar.update())
    pool.close()
    pool.join()
    print('All processes done.')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare cat dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lmdb-path', help='lmdb file path')
    parser.add_argument('--meta-file-path', help='meta file path')
    parser.add_argument('--out-dir', help='output directory of dataset')
    parser.add_argument(
        '--n-thread',
        nargs='?',
        default=8,
        type=int,
        help='thread number when using multiprocessing')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if not osp.exists(args.out_dir):
        os.makedirs(args.out_dir)
    _gt_dir = osp.join(args.out_dir, '_GT')
    gt_dir = osp.join(args.out_dir, 'GT')
    bix8_dir = osp.join(args.out_dir, 'BIx8_down')
    bix16_dir = osp.join(args.out_dir, 'BIx16_down')
    if not osp.exists(_gt_dir):
        os.makedirs(_gt_dir)
    if not osp.exists(gt_dir):
        os.makedirs(gt_dir)
    if not osp.exists(bix8_dir):
        os.makedirs(bix8_dir)
    if not osp.exists(bix16_dir):
        os.makedirs(bix16_dir)

    # export images
    export_images(args.lmdb_path, args.meta_file_path, _gt_dir)

    # downsample images
    downsample_images(args)
