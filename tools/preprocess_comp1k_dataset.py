import argparse
import math
import os.path as osp
import subprocess
from itertools import chain, repeat

import mmcv
import numpy as np
from PIL import Image


def fix_png_files(directory):
    """Fix png files in the target directory using pngfix.

    pngfix is a tool to fix PNG files. It's installed on Linux or MacOS by
    default.

    Args:
        directory (str): Directory to run pngfix.
    """
    subprocess.call(
        'pngfix --quiet --strip=color --prefix=fixed_ *.png',
        cwd=f'{directory}',
        shell=True)
    subprocess.call(
        'for fixed_f in fixed_*; do mv "$fixed_f" "${fixed_f:6}"; done',
        cwd=f'{directory}',
        shell=True)


def join_first_contain(directories, filename, data_root):
    """Join the first directory that contains the file.

    Args:
        directories (list[str]): Directories to search for the file.
        filename (str): The target filename.
        data_root (str): Root of the data path.
    """
    for directory in directories:
        cur_path = osp.join(directory, filename)
        if osp.exists(osp.join(data_root, cur_path)):
            return cur_path
    raise FileNotFoundError(f'Cannot find {filename} in dirs {directories}')


def get_data_info(args):
    """Function to process one piece of data.

    Args:
        args (tuple): Information needed to process one piece of data.

    Returns:
        dict: The processed data info.
    """
    name_with_postfix, source_bg_path, repeat_info, constant = args
    alpha, fg, alpha_path, fg_path = repeat_info
    data_root, composite, mode = constant

    if mode == 'training':
        dir_prefix = 'Training_set'
        trimap_dir = None
    elif mode == 'test':
        dir_prefix = 'Test_set'
        trimap_dir = 'Test_set/Adobe-licensed images/trimaps'
    else:
        raise KeyError(f'Unknown mode {mode}.')
    bg_path = osp.join(dir_prefix, 'bg',
                       name_with_postfix).replace('.jpg', '.png')
    merged_path = osp.join(dir_prefix, 'merged',
                           name_with_postfix).replace('.jpg', '.png')

    if not osp.exists(source_bg_path):
        raise FileNotFoundError(f'{source_bg_path} does not exist!')
    bg = Image.open(source_bg_path).convert('RGB')
    bw, bh = bg.size
    w, h = fg.size

    # rescale and crop bg
    wratio = float(w) / bw
    hratio = float(h) / bh
    ratio = wratio if wratio > hratio else hratio
    if ratio > 1:
        bg = bg.resize((math.ceil(bw * ratio), math.ceil(bh * ratio)),
                       Image.BICUBIC)
    bg = bg.crop((0, 0, w, h))

    # save cropped bg and merged
    mmcv.utils.mkdir_or_exist(osp.join(data_root, dir_prefix, 'bg'))
    bg.save(osp.join(data_root, bg_path), 'PNG')
    if composite:
        merged = (fg * alpha + bg * (1. - alpha)).astype(np.uint8)
        mmcv.utils.mkdir_or_exist(osp.join(data_root, dir_prefix, 'merged'))
        Image.fromarray(merged).save(osp.join(data_root, merged_path), 'PNG')

    data_info = dict()
    data_info['alpha_path'] = alpha_path
    data_info['fg_path'] = fg_path
    data_info['bg_path'] = bg_path
    data_info['merged_path'] = merged_path
    if trimap_dir is not None:
        trimap_path = osp.join(trimap_dir, name_with_postfix)
        trimap_full_path = osp.join(data_root, trimap_path)
        if not osp.exists(trimap_full_path):
            raise FileNotFoundError(f'{trimap_full_path} does not exist!')
        data_info['trimap_path'] = trimap_path
    return data_info


def generate_json(data_root, source_bg_dir, composite, nproc, mode):
    """Generate training json list or test json list.

    It should be noted except for `source_bg_dir`, other strings are incomplete
    relative path. When using these strings to read from or write to disk, a
    data_root is added to form a complete relative path.

    Args:
        data_root (str): path to Adobe composition-1k directory.
        source_bg_dir (str): source background directory.
        composite (bool): whether composite fg with bg and write to file.
        nproc (int): number of processers.
        mode (str): training or test mode.
    """

    if mode == 'training':
        dir_prefix = 'Training_set'
        fname_prefix = 'training'
        num_bg = 100  # each training fg is composited with 100 bg
        fg_dirs = [
            'Training_set/Adobe-licensed images/fg', 'Training_set/Other/fg'
        ]
        alpha_dirs = [
            'Training_set/Adobe-licensed images/alpha',
            'Training_set/Other/alpha'
        ]
    elif mode == 'test':
        dir_prefix = 'Test_set'
        fname_prefix = 'test'
        num_bg = 20  # each test fg is composited with 20 bg
        fg_dirs = ['Test_set/Adobe-licensed images/fg']
        alpha_dirs = ['Test_set/Adobe-licensed images/alpha']
    else:
        raise KeyError(f'Unknown mode {mode}.')
    fg_names = osp.join(dir_prefix, f'{fname_prefix}_fg_names.txt')
    bg_names = osp.join(dir_prefix, f'{fname_prefix}_bg_names.txt')
    save_json_path = f'{fname_prefix}_list.json'

    fg_names = open(osp.join(data_root, fg_names)).readlines()
    bg_names = open(osp.join(data_root, bg_names)).readlines()
    assert len(fg_names) * num_bg == len(bg_names)

    repeat_infos = []
    name_with_postfix = []
    # repeat fg and alpha num_bg time
    for fg_name in fg_names:
        fg_name = fg_name.strip()
        alpha_path = join_first_contain(alpha_dirs, fg_name, data_root)
        fg_path = join_first_contain(fg_dirs, fg_name, data_root)
        alpha_full_path = osp.join(data_root, alpha_path)
        fg_full_path = osp.join(data_root, fg_path)
        if not osp.exists(alpha_full_path):
            raise FileNotFoundError(f'{alpha_full_path} does not exist!')
        if not osp.exists(fg_full_path):
            raise FileNotFoundError(f'{fg_full_path} does not exist!')
        # to be consistent with DIM's composition code, use PIL to read images
        fg = Image.open(fg_full_path).convert('RGB')
        alpha = (
            np.array(Image.open(alpha_full_path).convert('RGB')) /
            255. if composite else None)
        repeat_infos.append((alpha, fg, alpha_path, fg_path))

        for bg_idx in range(num_bg):
            name_with_postfix.append(fg_name[:-4] + '_' + str(bg_idx) +
                                     fg_name[-4:])
    repeat_infos = chain.from_iterable(
        (repeat(repeat_info, num_bg) for repeat_info in repeat_infos))
    source_bg_paths = []
    for bg_name in bg_names:
        bg_name = bg_name.strip()
        # in coco_2017, image names do not begin with 'COCO_train2014_'
        if '2017' in source_bg_dir:
            bg_name = bg_name[15:]  # get rid of 'COCO_train2014_'
        source_bg_paths.append(osp.join(source_bg_dir, bg_name))
    constants = repeat((data_root, composite, mode), len(bg_names))

    data_infos = mmcv.track_parallel_progress(
        get_data_info,
        list(zip(name_with_postfix, source_bg_paths, repeat_infos, constants)),
        nproc)

    mmcv.dump(data_infos, osp.join(data_root, save_json_path))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare Adobe composition 1k dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data_root', help='Adobe composition 1k dataset root')
    parser.add_argument('coco_root', help='COCO2014 or COCO2017 dataset root')
    parser.add_argument('voc_root', help='VOCdevkit directory root')
    parser.add_argument(
        '--composite',
        action='store_true',
        help='whether to composite training foreground and background offline')
    parser.add_argument(
        '--nproc', type=int, default=4, help='number of processer')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if not osp.exists(args.data_root):
        raise FileNotFoundError(f'{args.data_root} does not exist!')
    if not osp.exists(args.coco_root):
        raise FileNotFoundError(f'{args.coco_root} does not exist!')
    if not osp.exists(args.voc_root):
        raise FileNotFoundError(f'{args.voc_root} does not exist!')

    data_root = args.data_root
    print('preparing training data...')
    if osp.exists(osp.join(args.coco_root, 'train2014')):
        train_source_bg_dir = osp.join(args.coco_root, 'train2014')
    elif osp.exists(osp.join(args.coco_root, 'train2017')):
        train_source_bg_dir = osp.join(args.coco_root, 'train2017')
    else:
        raise FileNotFoundError(
            f'Could not find train2014 or train2017 under {args.coco_root}')
    generate_json(data_root, train_source_bg_dir, args.composite, args.nproc,
                  'training')

    # remove the iCCP chunk from the PNG image to avoid unnecessary warning
    if args.composite:
        merged_dir = 'Training_set/merged'
        fix_png_files(osp.join(data_root, merged_dir))
    bg_dir = 'Training_set/bg'
    fix_png_files(osp.join(data_root, bg_dir))

    fg_dir = 'Test_set/Adobe-licensed images/fg'
    alpha_dir = 'Test_set/Adobe-licensed images/alpha'
    fix_png_files(osp.join(data_root, fg_dir))
    fix_png_files(osp.join(data_root, alpha_dir))

    print('\npreparing test data...')
    test_source_bg_dir = osp.join(args.voc_root, 'VOC2012/JPEGImages')
    generate_json(data_root, test_source_bg_dir, True, args.nproc, 'test')

    print('\nDone!')


if __name__ == '__main__':
    main()
