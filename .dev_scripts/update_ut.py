import os
import os.path as osp
from argparse import ArgumentParser
from fnmatch import fnmatch
from glob import glob

from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('--src', type=str, default='mmagic')
parser.add_argument('--dst', type=str, default='tests')
parser.add_argument(
    '--exclude',
    nargs='+',
    default=[
        'mmagic/.mim', 'mmagic/registry.py', 'mmagic/version.py',
        '__pycache__', '__init__', '**/__init__.py', '**/stylegan3_ops/*',
        '**/conv2d_gradfix.py', '**/grid_sample_gradfix.py', '**/misc.py',
        '**/upfirdn2d.py', '**/all_gather_layer.py', '**/typing.py'
    ])
args = parser.parse_args()


def check_exclude(fn):
    for pattern in args.exclude:
        if fnmatch(fn, pattern):
            return True
    return False


def update_ut():

    target_ut = []
    missing_ut = []
    blank_ut = []

    file_list = glob('mmagic/**/*.py', recursive=True)

    for f in tqdm(file_list):
        if check_exclude(f):
            continue

        if osp.splitext(osp.basename(f))[0] != '__init__':

            dirname = osp.dirname(f)
            dirname = dirname.replace('__', '')
            dirname = dirname.replace('mmagic', 'tests')
            dirname = dirname.replace('/', '/test_')
            os.makedirs(dirname, exist_ok=True)

            basename = osp.basename(f)
            basename = 'test_' + basename

            dst_path = osp.join(dirname, basename)
            target_ut.append(dst_path)
            if not osp.exists(dst_path):
                missing_ut.append(dst_path)
                fp = open(dst_path, 'a')
                fp.close()
            else:
                text_lines = open(dst_path, 'r').readlines()
                if len(text_lines) <= 3:
                    blank_ut.append(dst_path)

    existing_ut = glob('tests/test_*/**/*.py', recursive=True)
    additional_ut = list(set(existing_ut) - set(target_ut))

    if len(additional_ut) > 0:
        print('Additional UT:')
        for f in additional_ut:
            print(f)
    if len(missing_ut) > 0:
        print('Missing UT:')
        for f in missing_ut:
            print(f)
    if len(blank_ut) > 0:
        print('Blank UT:')
        for f in blank_ut:
            print(f)


if __name__ == '__main__':
    update_ut()
