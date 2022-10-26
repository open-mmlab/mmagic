# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import re
import subprocess

import mmcv
import numpy as np
from PIL import Image
from pymatting import estimate_foreground_ml, load_image


def fix_png_file(filename, folder):
    """Fix png files in the target filename using pngfix.

    pngfix is a tool to fix PNG files. It's installed on Linux or MacOS by
    default.

    Args:
        filename (str): png file to run pngfix.
    """
    subprocess.call(
        f'pngfix --quiet --strip=color --prefix=fixed_ "{filename}"',
        cwd=f'{folder}',
        shell=True)
    subprocess.call(
        f'mv "fixed_{filename}" "{filename}"', cwd=f'{folder}', shell=True)


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


class ExtendFg:

    def __init__(self, data_root, fg_dirs, alpha_dirs) -> None:
        self.data_root = data_root
        self.fg_dirs = fg_dirs
        self.alpha_dirs = alpha_dirs

    def extend(self, fg_name):
        fg_name = fg_name.strip()
        alpha_path = join_first_contain(self.alpha_dirs, fg_name,
                                        self.data_root)
        fg_path = join_first_contain(self.fg_dirs, fg_name, self.data_root)
        alpha_path = osp.join(self.data_root, alpha_path)
        fg_path = osp.join(self.data_root, fg_path)
        extended_path = re.sub('/fg/', '/fg_extended/', fg_path)
        extended_path = extended_path.replace('jpg', 'png')
        if not osp.exists(alpha_path):
            raise FileNotFoundError(f'{alpha_path} does not exist!')
        if not osp.exists(fg_path):
            raise FileNotFoundError(f'{fg_path} does not exist!')

        image = load_image(fg_path, 'RGB')
        alpha = load_image(alpha_path, 'GRAY')
        F = estimate_foreground_ml(image, alpha, return_background=False)
        fg = Image.fromarray(np.uint8(F * 255))
        fg.save(extended_path)
        fix_png_file(osp.basename(extended_path), osp.dirname(extended_path))
        data_info = dict()
        data_info['alpha_path'] = alpha_path
        data_info['fg_path'] = extended_path
        return data_info


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare Adobe composition 1k dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data_root', help='Adobe composition 1k dataset root')
    parser.add_argument(
        '--nproc', type=int, default=4, help='number of processor')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if not osp.exists(args.data_root):
        raise FileNotFoundError(f'{args.data_root} does not exist!')

    data_root = args.data_root

    print('preparing training data...')

    dir_prefix = 'Training_set'
    fname_prefix = 'training'
    fg_dirs = [
        'Training_set/Adobe-licensed images/fg', 'Training_set/Other/fg'
    ]
    alpha_dirs = [
        'Training_set/Adobe-licensed images/alpha', 'Training_set/Other/alpha'
    ]
    extended_dirs = [
        'Training_set/Adobe-licensed images/fg_extended',
        'Training_set/Other/fg_extended'
    ]
    for p in extended_dirs:
        p = osp.join(data_root, p)
        os.makedirs(p, exist_ok=True)

    fg_names = osp.join(dir_prefix, f'{fname_prefix}_fg_names.txt')
    save_json_path = f'{fname_prefix}_list_fba.json'
    fg_names = open(osp.join(data_root, fg_names)).readlines()
    fg_iter = iter(fg_names)

    extend_fg = ExtendFg(data_root, fg_dirs, alpha_dirs)
    data_infos = mmcv.track_parallel_progress(extend_fg.extend, list(fg_iter),
                                              args.nproc)
    mmcv.dump(data_infos, osp.join(data_root, save_json_path))

    print('train done')


if __name__ == '__main__':
    main()
