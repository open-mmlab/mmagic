# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import math
import os
import os.path as osp
import sys
from multiprocessing import Pool

import cv2
import lmdb
import mmcv
import mmengine
import numpy as np
from skimage import img_as_float
from skimage.io import imread, imsave

from mmagic.datasets.transforms import MATLABLikeResize, blur_kernels
from mmagic.utils import modify_args


def make_lmdb(mode,
              data_path,
              lmdb_path,
              train_list,
              batch=5000,
              compress_level=1):
    """Create lmdb for the Vimeo90K dataset.

    Contents of lmdb. The file structure is:
    example.lmdb
    ├── data.mdb
    ├── lock.mdb
    ├── meta_info.txt

    The data.mdb and lock.mdb are standard lmdb files and you can refer to
    https://lmdb.readthedocs.io/en/release/ for more details.

    The meta_info.txt is a specified txt file to record the meta information
    of our datasets. It will be automatically created when preparing
    datasets by our provided dataset tools.
    Each line in the txt file records 1)image name (with extension),
    2)image shape, and 3)compression level, separated by a white space.

    For example, the meta information could be:
    `000_00000000.png (720,1280,3) 1`, which means:
    1) image name (with extension): 000_00000000.png;
    2) image shape: (720,1280,3);
    3) compression level: 1

    We use the image name without extension as the lmdb key.

    Args:
        mode (str): Dataset mode. 'gt' or 'lq'.
        data_path (str): Data path for reading images.
        lmdb_path (str): Lmdb save path.
        train_list (str): Train list path for Vimeo90K datasets.
        batch (int): After processing batch images, lmdb commits.
            Default: 5000.
        compress_level (int): Compress level when encoding images. Default: 1.
    """

    print(f'Create lmdb for {data_path}, save to {lmdb_path}...')
    if mode == 'gt':
        h_dst, w_dst = 256, 448
    else:
        h_dst, w_dst = 64, 112

    if osp.exists(lmdb_path):
        print(f'Folder {lmdb_path} already exists. Exit.')
        sys.exit(1)

    print('Reading image path list ...')
    with open(train_list) as f:
        train_list = [line.strip() for line in f]

    all_img_list = []
    keys = []
    for line in train_list:
        folder, sub_folder = line.split('/')
        for j in range(1, 8):
            all_img_list.append(
                osp.join(data_path, folder, sub_folder, f'im{j}.png'))
            keys.append('{}_{}_{}'.format(folder, sub_folder, j))
    all_img_list = sorted(all_img_list)
    keys = sorted(keys)

    if mode == 'gt':  # only read the 4th frame for the gt mode
        print('Only keep the 4th frame for gt mode.')
        all_img_list = [v for v in all_img_list if v.endswith('im4.png')]
        keys = [v for v in keys if v.endswith('_4')]

    # create lmdb environment
    # obtain data size for one image
    img = mmcv.imread(osp.join(data_path, all_img_list[0]), flag='unchanged')
    _, img_byte = cv2.imencode('.png', img,
                               [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
    data_size_per_img = img_byte.nbytes
    print('Data size per image is: ', data_size_per_img)
    data_size = data_size_per_img * len(all_img_list)
    env = lmdb.open(lmdb_path, map_size=data_size * 10)

    # write data to lmdb
    pbar = mmengine.ProgressBar(len(all_img_list))
    txn = env.begin(write=True)
    txt_file = open(osp.join(lmdb_path, 'meta_info.txt'), 'w')
    for idx, (path, key) in enumerate(zip(all_img_list, keys)):
        pbar.update()
        key_byte = key.encode('ascii')
        img = mmcv.imread(osp.join(data_path, path), flag='unchanged')
        h, w, c = img.shape
        _, img_byte = cv2.imencode(
            '.png', img, [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
        assert h == h_dst and w == w_dst and c == 3, (
            f'Wrong shape ({h, w}), should be ({h_dst, w_dst}).')
        txn.put(key_byte, img_byte)
        # write meta information
        txt_file.write(f'{key}.png ({h},{w},{c}) {compress_level}\n')
        if idx % batch == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()
    env.close()
    txt_file.close()
    print('\nFinish writing lmdb.')


def generate_anno_file(clip_list, file_name='meta_info_Vimeo90K_GT.txt'):
    """Generate anno file for Vimeo90K datasets from the official clip list.

    Args:
        clip_list (str): Clip list path for Vimeo90K datasets.
        file_name (str): Saved file name. Default: 'meta_info_Vimeo90K_GT.txt'.
    """

    print(f'Generate annotation files {file_name}...')
    with open(clip_list) as f:
        lines = [line.rstrip() for line in f]
    txt_file = osp.join(osp.dirname(clip_list), file_name)
    with open(txt_file, 'w') as f:
        for line in lines:
            f.write(f'{line} 7 (256, 448, 3)\n')


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


def mesh_grid(kernel_size):
    """Generate the mesh grid, centering at zero.

    Args:
        kernel_size (int): The size of the kernel.

    Returns:
        xy_grid (np.ndarray): stacked xy coordinates with shape
            (kernel_size, kernel_size, 2).
    """
    range_ = np.arange(-(kernel_size - 1.) / 2., (kernel_size - 1.) / 2. + 1.)
    x_grid, y_grid = np.meshgrid(range_, range_)
    xy_grid = np.hstack((x_grid.reshape((kernel_size * kernel_size, 1)),
                         y_grid.reshape(kernel_size * kernel_size,
                                        1))).reshape(kernel_size, kernel_size,
                                                     2)

    return xy_grid


def bd_downsample(img_path, output_path, sigma=1.6, scale=4):
    """Downsampling using BD degradation(Gaussian blurring and downsampling).

    Args:
        img_path (str): Input image path.
        output_path (str): Output image path.
        sigma (float): The sigma of Gaussian blurring kernel. Default: 1.6.
        scale (int): The scale factor of the downsampling. Default: 4.
    """

    # Gaussian blurring
    kernelsize = math.ceil(sigma * 3) * 2 + 2
    kernel = blur_kernels.bivariate_gaussian(
        kernelsize, sigma, grid=mesh_grid(kernelsize))
    img = cv2.imread(img_path)
    img = img_as_float(img)
    output = cv2.filter2D(
        img,
        -1,
        kernel,
        anchor=((kernelsize - 1) // 2, (kernelsize - 1) // 2),
        borderType=cv2.BORDER_REPLICATE)

    # downsampling
    output = output[int(scale / 2) - 1:-int(scale / 2) + 1:scale,
                    int(scale / 2) - 1:-int(scale / 2) + 1:scale, :]

    output = np.clip(output, 0.0, 1.0) * 255
    output = output.astype(np.float32)
    output = np.floor(output + 0.5)
    cv2.imwrite(output_path, output)


def worker(clip_path, args):
    """Worker for each process.

    Args:
        clip_name (str): Path of the clip.

    Returns:
        process_info (str): Process information displayed in progress bar.
    """

    gt_dir = osp.join(args.data_root, 'GT', clip_path)
    bi_dir = osp.join(args.data_root, 'BIx4', clip_path)
    bd_dir = osp.join(args.data_root, 'BDx4', clip_path)
    mmengine.utils.mkdir_or_exist(bi_dir)
    mmengine.utils.mkdir_or_exist(bd_dir)

    img_list = sorted(os.listdir(gt_dir))
    for img in img_list:
        imresize(osp.join(gt_dir, img), osp.join(bi_dir, img), scale=1 / 4)
        bd_downsample(osp.join(gt_dir, img), osp.join(bd_dir, img))

    process_info = f'Processing {clip_path} ...'
    return process_info


def downsample_images(args):
    """Downsample images."""

    clip_list = []
    gt_dir = osp.join(args.data_root, 'GT')
    sequence_list = sorted(os.listdir(gt_dir))
    for sequence in sequence_list:
        sequence_root = osp.join(gt_dir, sequence)
        clip_list.extend(
            [osp.join(sequence, i) for i in sorted(os.listdir(sequence_root))])

    prog_bar = mmengine.ProgressBar(len(clip_list))
    pool = Pool(args.n_thread)
    for path in clip_list:
        pool.apply_async(
            worker, args=(path, args), callback=lambda arg: prog_bar.update())
    pool.close()
    pool.join()
    print('All processes done.')


def parse_args():
    modify_args()
    parser = argparse.ArgumentParser(
        description='Preprocess Vimeo90K datasets',
        epilog='You can download the Vimeo90K dataset '
        'from：http://toflow.csail.mit.edu/')
    parser.add_argument('--data-root', help='dataset root')
    parser.add_argument(
        '--n-thread',
        nargs='?',
        default=8,
        type=int,
        help='thread number when using multiprocessing')
    parser.add_argument(
        '--train_list',
        default=None,
        help='official training list path for Vimeo90K')
    parser.add_argument('--gt-path', default=None, help='GT path for Vimeo90K')
    parser.add_argument('--lq-path', default=None, help='LQ path for Vimeo90K')
    parser.add_argument(
        '--make-lmdb', action='store_true', help='create lmdb files')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # generate BIx4 and BDx4
    downsample_images(args)

    # generate image list anno file
    generate_anno_file(
        osp.join(args.data_root, 'sep_trainlist.txt'),
        'meta_info_Vimeo90K_train_GT.txt')
    generate_anno_file(
        osp.join(args.data_root, 'sep_testlist.txt'),
        'meta_info_Vimeo90K_test_GT.txt')

    # create lmdb files
    if args.make_lmdb:
        if args.gt_path is None or args.lq_path is None:
            raise ValueError('gt_path and lq_path cannot be None when '
                             'when creating lmdb files.')
        # create lmdb for gt
        lmdb_path = osp.join(
            osp.dirname(args.gt_path), 'vimeo90k_train_GT.lmdb')
        make_lmdb(
            mode='gt',
            data_path=args.gt_path,
            lmdb_path=lmdb_path,
            train_list=args.train_list)
        # create lmdb for lq
        lmdb_path = osp.join(
            osp.dirname(args.lq_path), 'vimeo90k_train_LR7frames.lmdb')
        make_lmdb(
            mode='lq',
            data_path=args.lq_path,
            lmdb_path=lmdb_path,
            train_list=args.train_list)
