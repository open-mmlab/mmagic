import argparse
import os
import os.path as osp
import re
from multiprocessing import Pool

import numpy as np
from PIL import Image
from preprocess_comp1k_dataset import join_first_contain
from pymatting import estimate_foreground_ml, load_image
from tqdm import tqdm


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
        alpha_full_path = osp.join(self.data_root, alpha_path)
        fg_full_path = osp.join(self.data_root, fg_path)
        extended_path = re.sub('/fg/', '/fg_extended/', fg_full_path)
        extended_path = extended_path.replace('jpg', 'png')
        if not osp.exists(alpha_full_path):
            raise FileNotFoundError(f'{alpha_full_path} does not exist!')
        if not osp.exists(fg_full_path):
            raise FileNotFoundError(f'{fg_full_path} does not exist!')
        image = load_image(fg_full_path, 'RGB')
        alpha = load_image(alpha_full_path, 'GRAY')
        F = estimate_foreground_ml(image, alpha, return_background=False)
        fg = Image.fromarray(np.uint8(F * 255))
        fg.save(extended_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare Adobe composition 1k dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data_root', help='Adobe composition 1k dataset root')
    parser.add_argument('coco_root', help='COCO2014 or COCO2017 dataset root')
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
    fg_names = open(osp.join(data_root, fg_names)).readlines()
    fg_iter = iter(fg_names)
    num = len(fg_names)

    extend_fg = ExtendFg(data_root, fg_dirs, alpha_dirs)
    with Pool(processes=args.nproc) as p:
        with tqdm(total=num) as pbar:
            for i, _ in tqdm(
                    enumerate(p.imap_unordered(extend_fg.extend, fg_iter))):
                pbar.update()

    print('train done')


if __name__ == '__main__':
    main()
