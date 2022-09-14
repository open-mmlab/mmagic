# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import sys

import mmengine
from mmengine import DictAction
from torchvision import utils

# yapf: disable
sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))  # isort:skip  # noqa

from mmedit.apis import init_model, sample_unconditional_model  # isort:skip  # noqa
# yapf: enable


def parse_args():
    parser = argparse.ArgumentParser(description='Generation demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--save-path',
        type=str,
        default='./work_dirs/demos/unconditional_samples.png',
        help='path to save unconditional samples')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CUDA device id')

    # args for inference/sampling
    parser.add_argument(
        '--num-batches', type=int, default=4, help='Batch size in inference')
    parser.add_argument(
        '--num-samples',
        type=int,
        default=12,
        help='The total number of samples')
    parser.add_argument(
        '--sample-model',
        type=str,
        default='ema',
        help='Which model to use for sampling')
    parser.add_argument(
        '--sample-cfg',
        nargs='+',
        action=DictAction,
        help='Other customized kwargs for sampling function')

    # args for image grid
    parser.add_argument(
        '--padding', type=int, default=0, help='Padding in the image grid.')
    parser.add_argument(
        '--nrow',
        type=int,
        default=6,
        help='Number of images displayed in each row of the grid')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model = init_model(
        args.config, checkpoint=args.checkpoint, device=args.device)

    if args.sample_cfg is None:
        args.sample_cfg = dict()

    results = sample_unconditional_model(model, args.num_samples,
                                         args.num_batches, args.sample_model,
                                         **args.sample_cfg)
    results = (results[:, [2, 1, 0]] + 1.) / 2.

    # save images
    mmengine.mkdir_or_exist(os.path.dirname(args.save_path))
    utils.save_image(
        results, args.save_path, nrow=args.nrow, padding=args.padding)


if __name__ == '__main__':
    main()
