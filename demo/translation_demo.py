# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import sys

import mmengine
from mmengine import DictAction
from torchvision import utils

# yapf: disable
sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))  # isort:skip  # noqa

from mmedit.apis import init_model, sample_img2img_model  # isort:skip  # noqa
# yapf: enable


def parse_args():
    parser = argparse.ArgumentParser(description='Translation demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('image_path', help='Image file path')
    parser.add_argument(
        '--target-domain', type=str, default=None, help='Desired image domain')
    parser.add_argument(
        '--save-path',
        type=str,
        default='./work_dirs/demos/translation_sample.png',
        help='path to save translation sample')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CUDA device id')
    # args for inference/sampling
    parser.add_argument(
        '--sample-cfg',
        nargs='+',
        action=DictAction,
        help='Other customized kwargs for sampling function')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model = init_model(
        args.config, checkpoint=args.checkpoint, device=args.device)

    if args.sample_cfg is None:
        args.sample_cfg = dict()

    results = sample_img2img_model(model, args.image_path, args.target_domain,
                                   **args.sample_cfg)
    results = (results[:, [2, 1, 0]] + 1.) / 2.

    # save images
    mmengine.mkdir_or_exist(os.path.dirname(args.save_path))
    utils.save_image(results, args.save_path)


if __name__ == '__main__':
    main()
