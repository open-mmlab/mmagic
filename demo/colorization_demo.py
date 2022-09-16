# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import mmcv
import torch

from mmedit.apis import colorization_inference, init_model
from mmedit.utils import modify_args


def parse_args():
    modify_args()
    parser = argparse.ArgumentParser(description='Colorzation demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('img_path', help='path to input image file')
    parser.add_argument('bbox_path', help='path to input image bbox file')
    parser.add_argument('save_path', help='path to save generation result')
    parser.add_argument(
        '--unpaired-path', default=None, help='path to unpaired image file')
    parser.add_argument(
        '--imshow', action='store_true', help='whether show image with opencv')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.device < 0 or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', args.device)

    #
    model = init_model(args.config, device=device)
    output = colorization_inference(model, args.img_path, args.bbox_path)

    if args.imshow:
        mmcv.imshow(output, 'predicted generation result')


if __name__ == '__main__':
    main()
