# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import mmcv
import torch

from mmedit.apis import init_model, restoration_inference
from mmedit.core import tensor2img
from mmedit.utils import modify_args


def parse_args():
    modify_args()
    parser = argparse.ArgumentParser(description='Restoration demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('img_path', help='path to input image file')
    parser.add_argument('save_path', help='path to save restoration result')
    parser.add_argument(
        '--imshow', action='store_true', help='whether show image with opencv')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    parser.add_argument(
        '--ref-path', default=None, help='path to reference image file')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if not os.path.isfile(args.img_path):
        raise ValueError('It seems that you did not input a valid '
                         '"image_path". Please double check your input, or '
                         'you may want to use "restoration_video_demo.py" '
                         'for video restoration.')
    if args.ref_path and not os.path.isfile(args.ref_path):
        raise ValueError('It seems that you did not input a valid '
                         '"ref_path". Please double check your input, or '
                         'you may want to use "ref_path=None" '
                         'for single restoration.')

    if args.device < 0 or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', args.device)

    model = init_model(args.config, args.checkpoint, device=device)

    if args.ref_path:  # Ref-SR
        output = restoration_inference(model, args.img_path, args.ref_path)
    else:  # SISR
        output = restoration_inference(model, args.img_path)
    output = tensor2img(output)

    mmcv.imwrite(output, args.save_path)
    if args.imshow:
        mmcv.imshow(output, 'predicted restoration result')


if __name__ == '__main__':
    main()
