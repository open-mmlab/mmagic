# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch
from mmengine import Config
from mmengine.registry import init_default_scope

from mmedit.registry import MODELS

try:
    from mmengine.analysis import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmengine >= 0.6.0')


def parse_args():
    parser = argparse.ArgumentParser(description='Get a editor complexity')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[3, 250, 250],
        help='input shape')
    parser.add_argument(
        '--activations',
        action='store_true',
        help='Whether to show the Activations')
    parser.add_argument(
        '--out-table',
        action='store_true',
        help='Whether to show the complexity table')
    parser.add_argument(
        '--out-arch',
        action='store_true',
        help='Whether to show the complexity arch')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    input_shape = tuple(args.shape)

    cfg = Config.fromfile(args.config)

    init_default_scope(cfg.get('default_scope', 'mmedit'))

    model = MODELS.build(cfg.model)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    analysis_results = get_model_complexity_info(model, input_shape)
    flops = analysis_results['flops_str']
    params = analysis_results['params_str']
    activations = analysis_results['activations_str']

    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}\n')
    if args.activations:
        print(f'Activations: {activations}\n{split_line}\n')
    if args.out_table:
        print(analysis_results['out_table'], '\n')
    if args.out_arch:
        print(analysis_results['out_arch'], '\n')

    if len(input_shape) == 4:
        print('!!!If your network computes N frames in one forward pass, you '
              'may want to divide the FLOPs by N to get the average FLOPs '
              'for each frame.')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
