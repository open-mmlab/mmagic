import argparse
import os.path as osp
import pickle

import mmcv
import numpy as np
import torch
from mmcv import Config, print_log

from mmedit.core.evaluation.inception_fid_metrics import extract_features
from mmedit.datasets import ImgFromFolderDataset, build_dataloader
from mmedit.models import InceptionV3

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pre-calculate inception data and save it in pkl file')
    parser.add_argument('imgsdir', type=str, help='the dir containing images')
    parser.add_argument('pklname', type=str, help='the name of inception pkl')
    parser.add_argument(
        '--pkl-dir',
        type=str,
        default='work_dirs/inception_pkl',
        help='path to save pkl file')
    parser.add_argument(
        '--pipeline-cfg',
        type=str,
        default=None,
        help=('config file containing dataset pipeline. If None, the default'
              ' pipeline will be adopted'))
    parser.add_argument(
        '--batch-size',
        type=int,
        default=25,
        help='batch size used in extracted features')
    parser.add_argument(
        '--num-samples',
        type=int,
        default=50000,
        help='the number of total samples')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # dataset pipeline
    if args.pipeline_cfg is not None:
        pipeline = Config.fromfile(args.pipeline_cfg)['pipeline']
    else:
        pipeline = [
            dict(type='LoadImageFromFile', key='img'),
            dict(
                type='Resize',
                keys=['img'],
                scale=(300, 300),
                keep_ratio=False,
            ),
            dict(
                type='Normalize',
                keys=['img'],
                mean=[127.5] * 3,
                std=[127.5] * 3,
                to_rgb=False),
            dict(type='Collect', keys=['img'], meta_keys=[]),
            dict(type='ImageToTensor', keys=['img'])
        ]

    mmcv.mkdir_or_exist(args.pkl_dir)

    # build dataloader
    dataset = ImgFromFolderDataset(
        args.imgsdir, pipeline, num_samples=args.num_samples)
    data_loader = build_dataloader(dataset, args.batch_size, 4, dist=False)

    # build inception network
    inception = InceptionV3([3], normalize_input=False).to(device)

    features = extract_features(data_loader, inception, device).numpy()

    # sanity check for the number of features
    assert features.shape[
        0] == args.num_samples, 'the number of features != num_samples'
    print_log(f'extracted {args.num_samples} features', 'mmedit')

    mean = np.mean(features, 0)
    cov = np.cov(features, rowvar=False)

    with open(osp.join(args.pkl_dir, args.pklname), 'wb') as f:
        pickle.dump(
            {
                'mean': mean,
                'cov': cov,
                'size': args.num_samples,
                'name': args.pklname
            }, f)
