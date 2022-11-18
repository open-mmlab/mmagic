# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch
from mmengine import Config

from mmedit.registry import MODELS
from mmedit.utils import register_all_modules

try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')


def parse_args():
    parser = argparse.ArgumentParser(description='Train a editor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[250, 250],
        help='input image size')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    elif len(args.shape) in [3, 4]:  # 4 for video inputs (t, c, h, w)
        input_shape = tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    register_all_modules()

    cfg = Config.fromfile(args.config)
    model = MODELS.build(cfg.model)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    elif hasattr(model, 'forward_tensor'):
        model.forward = model.forward_tensor
    # else:
    #     raise NotImplementedError(
    #         'FLOPs counter is currently not currently supported '
    #         f'with {model.__class__.__name__}')

    flops, params = get_model_complexity_info(model, input_shape)

    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    if len(input_shape) == 4:
        print('!!!If your network computes N frames in one forward pass, you '
              'may want to divide the FLOPs by N to get the average FLOPs '
              'for each frame.')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
