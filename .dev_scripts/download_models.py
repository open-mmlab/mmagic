#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.

import argparse
import os
import platform
import posixpath as osp  # Even on windows, use posixpath
import re
import subprocess
from collections import OrderedDict
from importlib.machinery import SourceFileLoader
from pathlib import Path

from modelindex import load

MMagic_ROOT = Path(__file__).absolute().parent.parent
DOWNLOAD_DIR = osp.join(MMagic_ROOT, 'work_dirs', 'download')
IS_WINDOWS = (platform.system() == 'Windows')


def additional_download(args):
    """Download additional weights file used in this repo, such as VGG."""

    url_path = [
        'https://www.adrianbulat.com/downloads/python-fan/2DFAN4-cd938726ad.zip',  # noqa
        'https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth',
        'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
    ]
    checkpoint_root = args.checkpoint_root
    if not checkpoint_root:
        checkpoint_root = DOWNLOAD_DIR
    for url in url_path:
        path = osp.join(checkpoint_root, 'hub', 'checkpoints',
                        osp.basename(url))
        print()
        print(f'download {path} from {url}')

        if IS_WINDOWS:
            if args.dry_run:
                print(f'Call  \'wget.download({url}, {path})\'')
            else:
                import wget
                wget.download(url, path)
        else:
            if args.dry_run:
                print(f'wget --no-check-certificate -N {url} {path}')
            else:
                os.system(f'wget --no-check-certificate -N {url} {path}')


def parse_args():
    parser = argparse.ArgumentParser(description='Download the checkpoints')
    parser.add_argument('--checkpoint-root', help='Checkpoint file root path.')
    parser.add_argument(
        '--models', nargs='+', type=str, help='Specify model names to run.')
    parser.add_argument(
        '--force',
        action='store_true',
        help='Whether force re-download the checkpoints.')
    parser.add_argument(
        '--model-list', type=str, help='Path of algorithm list to download')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only show download command but not run')

    args = parser.parse_args()
    return args


def download(args):
    model_index_file = MMagic_ROOT / 'model-index.yml'
    model_index = load(str(model_index_file))
    model_index.build_models_with_collections()
    models = OrderedDict({model.name: model for model in model_index.models})

    http_prefix_short = 'https://download.openmmlab.com/mmediting/'

    # load model list
    if args.model_list:
        file_list_path = args.model_list
        file_list = SourceFileLoader('model_list',
                                     file_list_path).load_module().model_list
    else:
        file_list = None

    if args.models:
        patterns = [re.compile(pattern) for pattern in args.models]
        filter_models = {}
        for k, v in models.items():
            if any([re.match(pattern, k) for pattern in patterns]):
                filter_models[k] = v
        if len(filter_models) == 0:
            print('No model found, please specify models in:')
            print('\n'.join(models.keys()))
            return
        models = filter_models

    checkpoint_root = args.checkpoint_root
    if not checkpoint_root:
        checkpoint_root = DOWNLOAD_DIR

    for model_info in models.values():

        if file_list is not None and model_info.name not in file_list:
            continue

        model_weight_url = model_info.weights

        if model_weight_url.startswith(http_prefix_short):
            model_name = model_weight_url[len(http_prefix_short):]
        elif model_weight_url == '':
            print(f'{model_info.Name} weight is missing')
            return None
        else:
            raise ValueError(f'Unknown url prefix. \'{model_weight_url}\'')

        model_name_split = model_name.split('/')
        if len(model_name_split) == 3:  # 'TASK/METHOD/MODEL.pth'
            # remove task name
            model_name = osp.join(*model_name_split[1:-1])
        else:
            model_name = osp.join(*model_name_split[:-1])
        ckpt_name = model_weight_url.split('/')[-1]

        download_path = osp.join(checkpoint_root, model_name, ckpt_name)
        download_root = osp.join(checkpoint_root, model_name)

        if osp.exists(download_path):
            print(f'Already exists {download_path}')
            # do not delete when dry-run is true
            if args.force and not args.dry_run:
                print(f'Delete {download_path} to force re-download.')
                os.system(f'rm -rf {download_path}')
            else:
                continue
        try:
            cmd_str_list = [
                'wget', '-q', '--show-progress', '-p', download_root,
                model_weight_url
            ]

            if args.dry_run:
                print(' '.join(cmd_str_list))
            else:
                subprocess.run(cmd_str_list, check=True)
        except Exception:
            # for older version of wget
            cmd_str_list = ['wget', '-P', download_root, model_weight_url]
            if args.dry_run:
                print(' '.join(cmd_str_list))
            else:
                subprocess.run(cmd_str_list, check=True)


if __name__ == '__main__':
    args = parse_args()
    download(args)
    additional_download(args)
