import argparse
import os
import os.path as osp
import re
import sys
import scipy.io as io
from multiprocessing import Pool

import cv2
import lmdb
import mmcv
import numpy as np


def main(args):
    """A multi-thread tool to generate dataset pair for Blind Super-Resolution

    It is built based on preprocess_div2k_dataset.py

    opt (dict): Configuration dict. It contains:
        n_thread (int): Thread number.
        compression_level (int):  CV_IMWRITE_PNG_COMPRESSION from 0 to 9.
            A higher value means a smaller size and longer compression time.
            Use 0 for faster CPU decompression. Default: 3, same in cv2.

        input_folder (str): Path to the input folder.
        save_folder (str): Path to save folder.
        crop_size (int): Crop size.

        kernel_size (int): size of blur kernel.
        isotropic (boolean): True for isotropic gaussian blur kernel, False for anisotropic.
        random_disturb (boolean): True for adding disturb term to anisotropic kernel.
        kernel_random (boolean): True for using same kernels for each image,
            False for using different kernel for each image.
        kernel_sigma_min (float): If kernel_random == True, it control the lower bound of sigma of gaussian filter.
        kernel_sigma_max (float): If kernel_random == True, it control the upper bound of sigma of gaussian filter.

        step (int): Step for overlapped sliding window.
        thresh_size (int): Threshold size. Patches whose size is lower
            than thresh_size will be dropped.

    Usage:
        For each folder, run this script.
        Typically, there is only one folder to be processed for DIV2K dataset.
            DIV2K_train_HR
        After process, each sub_folder should have the same number of sub-images and get folders like:
            DIV2K_train_HR_sub
            DIV2K_train_LQ_sub/X2
            DIV2K_train_LQ_sub/X3
            DIV2K_train_LQ_sub/X4
            DIV2K_train_kernels
            kernel_parameters.txt
        Typically, kernels are stored in the same format as DIV2KRK kernels.
        In kernel_parameters, it contains all parameters user defined for kernels in DIV2K_train_kernels
        Remember to modify opt configurations according to your settings.
    """

    opt = {}
    opt['n_thread'] = args.n_thread
    opt['compression_level'] = args.compression_level

    # HR images related
    opt['input_folder'] = osp.join(args.data_root, 'DIV2K_train_HR')
    opt['save_folder'] = args.data_root
    opt['crop_size'] = args.crop_size
    opt['step'] = args.step
    opt['thresh_size'] = args.thresh_size
    opt['scale'] = args.scale

    # kernel related
    opt['isotropic'] = args.isotropic
    opt['random_disturb'] = args.random_disturb
    opt['kernel_size'] = args.kernel_size
    opt['kernel_random'] = args.kernel_random
    opt['kernel_sigma'] = args.kernel_sigma
    opt['kernel_sigma_min'] = args.kernel_sigma_min
    opt['kernel_sigma_max'] = args.kernel_sigma_max

    handle_subimages(opt)


def handle_subimages(opt):
    """Crop, Blur, Down-sample images

    Args:
        opt (dict): Configuration dict. It contains:
            n_thread (int): Thread number.
            compression_level (int):  CV_IMWRITE_PNG_COMPRESSION from 0 to 9.
                A higher value means a smaller size and longer compression time.
                Use 0 for faster CPU decompression. Default: 3, same in cv2.

            input_folder (str): Path to the input folder.
            save_folder (str): Path to save folder.
            crop_size (int): Crop size.

            isotropic (boolean): True for isotropic gaussian blur kernel, False for anisotropic.
            kernel_size (int): size of blur kernel.
            kernel_random (boolean): True for using same kernels, False for using different kernel for each image.
            kernel_sigma_min (float): If kernel_random == True, it control the lower bound of sigma of gaussian filter.
            kernel_sigma_max (float): If kernel_random == True, it control the upper bound of sigma of gaussian filter.

            step (int): Step for overlapped sliding window.
            thresh_size (int): Threshold size. Patches whose size is lower
                than thresh_size will be dropped.
    """
    input_folder = opt['input_folder']

    # hr path
    hr_folder = osp.join(opt['save_folder'], 'DIV2K_train_HR_sub')

    # lq path
    lq_folder = osp.join(opt['save_folder'], 'DIV2K_train_LQ_sub', 'x{scale}'.format(scale=opt['scale']))

    # kernel path
    kernel_folder = osp.join(opt['save_folder'], 'DIV2K_train_kernel')

    # write description about kernel
    f = open(osp.join(opt['save_folder'], 'kernel_parameters.txt'), 'w')
    f.write(f'kernel size:{opt["kernel_size"]} \n'
            f'isotropic:{opt["isotropic"]} \n'
            f'random:{opt["kernel_random"]} \n'
            f'sigma:{opt["kernel_sigma"]}, it works when random == False\n'
            f'min sigma:{opt["kernel_sigma_min"]} it works when random == True \n'
            f'max sigma:{opt["kernel_sigma_max"]} it works when random == True \n'

            f'using random disturb term in anisotropic kernel:{opt["random_disturb"]} \n')
    f.close()

    folders = [hr_folder, lq_folder, kernel_folder]

    for folder in folders:
        if not osp.exists(folder):
            os.makedirs(folder)
            print(f'mkdir {folder} ...')
        else:
            print(f'Folder {folder} already exists. Exit.')
            sys.exit(1)

    opt['hr_folder'] = hr_folder
    opt['lq_folder'] = lq_folder
    opt['kernel_folder'] = kernel_folder

    img_list = list(mmcv.scandir(input_folder))
    img_list = [osp.join(input_folder, v) for v in img_list]

    prog_bar = mmcv.ProgressBar(len(img_list))
    pool = Pool(opt['n_thread'])
    for path in img_list:
        pool.apply_async(
            worker, args=(path, opt), callback=lambda arg: prog_bar.update())
    pool.close()
    pool.join()
    print('All processes done.')


def generate_kernels(kernel_size,
                     random,
                     sigma,
                     sigma_min,
                     sigma_max,
                     isotropic_rate,
                     random_disturb):
    l = int((kernel_size - 1) / 2)
    K = np.zeros([kernel_size, kernel_size])

    if isotropic_rate:
        sigma = np.random.uniform(sigma_min, sigma_max) if random else sigma
        for i in range(-l, l + 1):
            for j in range(-l, l + 1):
                K[i + l, j + l] = np.exp(-0.5 * (i ** 2 + j ** 2) / sigma ** 2)
        K = K / np.sum(K)
    else:
        sigma_x = np.random.uniform(sigma_min, sigma_max) if random else sigma
        sigma_y = np.random.uniform(sigma_min, sigma_max) if random else sigma

        # radians
        radians = np.random.uniform(-np.pi, np.pi)

        # beta
        beta = 1

        # R
        R = np.zeros([2, 2])
        R[0, 0] = np.cos(radians)
        R[0, 1] = -np.sin(radians)
        R[1, 0] = np.sin(radians)
        R[1, 1] = np.cos(radians)

        # V
        V = np.zeros([2, 2])
        V[0, 0] = sigma_x
        V[1, 1] = sigma_y

        # covariance matrix
        U = np.matmul(np.matmul(R, V), R.T)

        # kernel
        for i in range(-l, l + 1):
            for j in range(-l, l + 1):
                C = np.array([i, j])
                c = np.matmul(np.matmul(C.T, np.linalg.inv(U)), C)
                K[i + l, j + l] = np.exp(-0.5 * c ** beta)

        # disturb
        if random_disturb:
            K = K + np.random.uniform(0, 0.25) * K
        K = K / np.sum(K)

    return K


def worker(path, opt):
    """Worker for each process.

    Args:
        path (str): Image path.
        opt (dict): Configuration dict. It contains:
            crop_size (int): Crop size.
            isotropic(boolean): True for isotropic gaussian filter, False for anisotropic gaussian filter.
            kernel_size (int): size of blur kernel.
            kernel_random (boolean): True for using same kernels, False for using different kernel for each image.
            kernel_sigma (float): work when kernel_random == False, it is the value of sigma of gaussian filter.
            kernel_sigma_min (float): work when kernel_random == True,
                it control the lower bound of sigma of gaussian filter.
            kernel_sigma_max (float): work when kernel_random == True,
                it control the upper bound of sigma of gaussian filter.


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
    img_name, extension = osp.splitext(osp.basename(path))

    # # remove the x2, x3, x4 and x8 in the filename for DIV2K
    # img_name = re.sub('x[2348]', '', img_name)
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
            # cropped hr image
            cropped_img = img[x:x + crop_size, y:y + crop_size, ...]

            # blur kernel

            kernel = generate_kernels(kernel_size=opt['kernel_size'],
                                      random=opt['kernel_random'],
                                      sigma=opt['kernel_sigma'],
                                      sigma_min=opt['kernel_sigma_min'],
                                      sigma_max=opt['kernel_sigma_max'],
                                      isotropic_rate=opt['isotropic'],
                                      random_disturb=opt['random_disturb']).reshape(opt['kernel_size'],
                                                                                    opt['kernel_size'])
            # blur image
            blur_img = cv2.filter2D(cropped_img, -1, kernel)

            # down-sample image
            lq_img = cv2.resize(blur_img, (int(crop_size / opt['scale']), int(crop_size / opt['scale'])),
                                interpolation=cv2.INTER_CUBIC)

            # save Low-Quality sub image
            cv2.imwrite(
                osp.join(opt['lq_folder'],
                         f'{img_name}_s{index:03d}{extension}'), lq_img,
                [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])

            # save Ground-Truth sub image
            cv2.imwrite(
                osp.join(opt['hr_folder'],
                         f'{img_name}_s{index:03d}{extension}'), cropped_img,
                [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])

            # save kernel
            io.savemat(
                osp.join(opt['kernel_folder'],
                         f'{img_name}_s{index:03d}.mat'),
                {'Kernel': kernel})

    process_info = f'Processing {img_name} ...'
    return process_info


def make_lmdb_for_div2k(data_root):
    """Create lmdb files for DIV2K dataset.

    Args:
        data_root (str): Data root path.

    Usage:
        Typically, there are four folders to be processed for DIV2K dataset.
            DIV2K_train_HR_sub
            DIV2K_train_LQ_sub/X2
            DIV2K_train_LQ_sub/X3
            DIV2K_train_LQ_sub/X4
        Remember to modify opt configurations according to your settings.
    """

    folder_paths = [
        osp.join(data_root, 'DIV2K_train_HR_sub'),
        osp.join(data_root, 'DIV2K_train_LQ_sub/X2'),
        osp.join(data_root, 'DIV2K_train_LQ_sub/X3'),
        osp.join(data_root, 'DIV2K_train_LQ_sub/X4')
    ]
    lmdb_paths = [
        osp.join(data_root, 'DIV2K_train_HR_sub.lmdb'),
        osp.join(data_root, 'DIV2K_train_LQ_X2_sub.lmdb'),
        osp.join(data_root, 'DIV2K_train_LQ_X3_sub.lmdb'),
        osp.join(data_root, 'DIV2K_train_LQ_X4_sub.lmdb')
    ]

    for folder_path, lmdb_path in zip(folder_paths, lmdb_paths):
        img_path_list, keys = prepare_keys_div2k(folder_path)
        make_lmdb(folder_path, lmdb_path, img_path_list, keys)


def prepare_keys_div2k(folder_path):
    """Prepare image path list and keys for DIV2K dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(
        list(mmcv.scandir(folder_path, suffix='png', recursive=False)))
    keys = [img_path.split('.png')[0] for img_path in sorted(img_path_list)]

    return img_path_list, keys


def make_lmdb(data_path,
              lmdb_path,
              img_path_list,
              keys,
              batch=5000,
              compress_level=1,
              multiprocessing_read=False,
              n_thread=40):
    """Make lmdb.

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

    If `multiprocessing_read` is True, it will read all the images to memory
    using multiprocessing. Thus, your server needs to have enough memory.

    Args:
        data_path (str): Data path for reading images.
        lmdb_path (str): Lmdb save path.
        img_path_list (str): Image path list.
        keys (str): Used for lmdb keys.
        batch (int): After processing batch images, lmdb commits.
            Default: 5000.
        compress_level (int): Compress level when encoding images. Default: 1.
        multiprocessing_read (bool): Whether use multiprocessing to read all
            the images to memory. Default: False.
        n_thread (int): For multiprocessing.
    """
    assert len(img_path_list) == len(keys), (
        'img_path_list and keys should have the same length, '
        f'but got {len(img_path_list)} and {len(keys)}')
    print(f'Create lmdb for {data_path}, save to {lmdb_path}...')
    print(f'Total images: {len(img_path_list)}')
    if not lmdb_path.endswith('.lmdb'):
        raise ValueError("lmdb_path must end with '.lmdb'.")
    if osp.exists(lmdb_path):
        print(f'Folder {lmdb_path} already exists. Exit.')
        sys.exit(1)

    if multiprocessing_read:
        # read all the images to memory (multiprocessing)
        dataset = {}  # use dict to keep the order for multiprocessing
        shapes = {}
        print(f'Read images with multiprocessing, #thread: {n_thread} ...')
        prog_bar = mmcv.ProgressBar(len(img_path_list))

        def callback(arg):
            """get the image data and update prog_bar."""
            key, dataset[key], shapes[key] = arg
            prog_bar.update()

        pool = Pool(n_thread)
        for path, key in zip(img_path_list, keys):
            pool.apply_async(
                read_img_worker,
                args=(osp.join(data_path, path), key, compress_level),
                callback=callback)
        pool.close()
        pool.join()
        print(f'Finish reading {len(img_path_list)} images.')

    # create lmdb environment
    # obtain data size for one image
    img = mmcv.imread(osp.join(data_path, img_path_list[0]), flag='unchanged')
    _, img_byte = cv2.imencode('.png', img,
                               [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
    data_size_per_img = img_byte.nbytes
    print('Data size per image is: ', data_size_per_img)
    data_size = data_size_per_img * len(img_path_list)
    env = lmdb.open(lmdb_path, map_size=data_size * 10)

    # write data to lmdb
    prog_bar = mmcv.ProgressBar(len(img_path_list))
    txn = env.begin(write=True)
    txt_file = open(osp.join(lmdb_path, 'meta_info.txt'), 'w')
    for idx, (path, key) in enumerate(zip(img_path_list, keys)):
        prog_bar.update()
        key_byte = key.encode('ascii')
        if multiprocessing_read:
            img_byte = dataset[key]
            h, w, c = shapes[key]
        else:
            _, img_byte, img_shape = read_img_worker(
                osp.join(data_path, path), key, compress_level)
            h, w, c = img_shape

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


def read_img_worker(path, key, compress_level):
    """Read image worker

    Args:
        path (str): Image path.
        key (str): Image key.
        compress_level (int): Compress level when encoding images.

    Returns:
        str: Image key.
        byte: Image byte.
        tuple[int]: Image shape.
    """
    img = mmcv.imread(path, flag='unchanged')
    if img.ndim == 2:
        h, w = img.shape
        c = 1
    else:
        h, w, c = img.shape
    _, img_byte = cv2.imencode('.png', img,
                               [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
    return (key, img_byte, (h, w, c))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare DIV2K dataset for Blind Super Resolution',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-root',
                        default='.',
                        help='dataset root')
    parser.add_argument(
        '--crop-size',
        nargs='?',
        default=256,
        help='cropped size for HR images')
    parser.add_argument(
        '--step', nargs='?', default=256, help='step size for HR images')
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
        '--scale',
        nargs='?',
        default=4,
        help='scale factor')

    parser.add_argument(
        '--kernel_size',
        nargs='?',
        default=31,
        help='size of kernel')

    parser.add_argument(
        '--isotropic',
        nargs='?',
        default=False,
        help='regular gaussian blur kernel or not')

    parser.add_argument(
        '--random_disturb',
        nargs='?',
        default=True,
        help='add disturb term in anisotropic kernel or not')

    parser.add_argument(
        '--kernel_random',
        nargs='?',
        default=True,
        help='whether different kernels for different images')

    parser.add_argument(
        '--kernel_sigma',
        nargs='?',
        default=1.0,
        help='sigma of kernel')

    parser.add_argument(
        '--kernel_sigma_min',
        nargs='?',
        default=0.6,
        help='min value of sigma of kernel,'
             'works when kernel_random is True')

    parser.add_argument(
        '--kernel_sigma_max',
        nargs='?',
        default=5.0,
        help='max value of sigma of kernel,'
             'works when kernel_random is True')

    parser.add_argument(
        '--n-thread',
        nargs='?',
        default=20,
        help='thread number when using multiprocessing')

    parser.add_argument(
        '--make-lmdb',
        default=False,
        action='store_true',
        help='whether to prepare lmdb files')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # handle images and store them along with kernels
    main(args)

    # prepare lmdb files if necessary
    if args.make_lmdb:
        make_lmdb_for_div2k(args.data_root)
