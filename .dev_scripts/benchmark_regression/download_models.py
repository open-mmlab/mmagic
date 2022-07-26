#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.

# This tool is used to download all models in configs.

import glob
import os
import platform
import posixpath as osp  # Even on windows, use posixpath
import re
import sys

MMEditing_ROOT = osp.dirname(osp.dirname(osp.dirname(__file__)))
DOWNLOAD_DIR = osp.join(MMEditing_ROOT, 'work_dirs', 'download')
IS_WINDOWS = (platform.system() == 'Windows')


def additional_download():
    """Download additional weights file used in this repo, such as VGG.
    """

    url_path = [
        'https://www.adrianbulat.com/downloads/python-fan/2DFAN4-cd938726ad.zip',  # noqa
        'https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth',
        'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
    ]

    for url in url_path:
        path = osp.join(DOWNLOAD_DIR, 'hub', 'checkpoints', osp.basename(url))
        print()
        print(f'download {path} from {url}')
        if IS_WINDOWS:
            import wget
            wget.download(url, path)
        else:
            os.system(f'wget -N {url} {path}')


def find_pth_files(file: str):
    """Find all strs of pth files from a file.

    Args:
        file (str): The original file (config or README).

    Returns:
        List[str]: List of pth files.
    """

    with open(file, 'r', encoding='utf-8') as f:
        data = f.read()

    if file.endswith('md'):
        pth_files = re.findall(r'\[model\]\((https://.*?\.pth)\)', data)
    else:
        pth_files = re.findall(r'=.?\'(https?://.*?\.pth)\'', data, re.S)

    return (pth_files)


def find_all_pth(md_file):
    """Find all pre-trained checkpoints of a method (pth).

    Args:
        md_file (str): Path to .md file.

    Returns:
        Bool: If the target .pth files are downloaded successfully.
    """

    md_file = md_file.replace(os.sep, '/')
    config_dir, _ = osp.split(md_file)
    files = os.listdir(config_dir)
    config_files = [
        osp.join(config_dir, file) for file in files if file.endswith('.py')
    ]
    all_files = config_files + [md_file]
    pth_files = []
    for file in all_files:
        sub_list = find_pth_files(file)
        if len(sub_list) > 0:
            pth_files.extend(sub_list)
    return pth_files


def download_pth(pth_files):
    """Download all pth files.

    Args:
        pth_files (List[str]): List of pth files.
    """

    # clear
    def clear_path(path: str):
        path = path.replace(' ', '')
        path = path.replace('\'', '')
        path = path.replace('\\', '')
        path = path.replace('+', '')
        path = path.replace('\n', '')
        return path

    pth_files = [clear_path(file) for file in pth_files]
    pth_files.sort()
    pth_files = set(pth_files)

    if not osp.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)

    for url in pth_files:
        path = osp.join(DOWNLOAD_DIR, 'hub', 'checkpoints', osp.basename(url))
        print()
        print(f'download {path} from {url}')
        if IS_WINDOWS:
            import wget
            wget.download(url, path)
        else:
            os.system(f'wget -N {url} {path}')


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        configs_root = osp.join(MMEditing_ROOT, 'configs')
        file_list = glob.glob(
            osp.join(configs_root, '**', '*README.md'), recursive=True)
        file_list.sort()
    else:
        file_list = [
            fn for fn in sys.argv[1:] if osp.basename(fn) == 'README.md'
        ]

    if not file_list:
        sys.exit(0)

    pth_files = []
    for fn in file_list:
        pth_files.extend(find_all_pth(fn))

    download_pth(pth_files)
    additional_download()
