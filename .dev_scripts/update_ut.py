import os
import os.path as osp
from argparse import ArgumentParser
from glob import glob

from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('--src', type=str, default='mmedit')
parser.add_argument('--dst', type=str, default='tests')
args = parser.parse_args()


def update_ut():

    folders = [f for f in os.listdir(args.src) if osp.isdir(f'mmedit/{f}')]
    target_ut = []
    missing_ut = []

    for subf in folders:
        if subf == '.mim' or subf == '__pycache__':
            continue

        file_list = glob(f'mmedit/{subf}/**/*.py', recursive=True)

        for f in tqdm(file_list, desc=f'mmedit/{subf}'):
            if osp.splitext(osp.basename(f))[0] != '__init__':

                dirname = osp.dirname(f)
                dirname = dirname.replace('__', '')
                dirname = dirname.replace('mmedit', 'tests')
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


if __name__ == '__main__':
    update_ut()
