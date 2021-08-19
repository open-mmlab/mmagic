# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import mmcv
import torch

from mmedit.apis import init_model, restoration_video_inference
from mmedit.core import tensor2img


def parse_args():
    parser = argparse.ArgumentParser(description='Restoration demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('input_dir', help='directory of the input video')
    parser.add_argument('output_dir', help='directory of the output video')
    parser.add_argument(
        '--start_idx',
        type=int,
        default=0,
        help='index corresponds to the first frame of the sequence')
    parser.add_argument(
        '--filename_tmpl',
        default='{:08d}.png',
        help='template of the file names')
    parser.add_argument(
        '--window_size',
        type=int,
        default=0,
        help='window size if sliding-window framework is used')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model = init_model(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))

    output = restoration_video_inference(model, args.input_dir,
                                         args.window_size, args.start_idx,
                                         args.filename_tmpl)
    for i in range(args.start_idx, args.start_idx + output.size(1)):
        output_i = output[:, i - args.start_idx, :, :, :]
        output_i = tensor2img(output_i)
        save_path_i = f'{args.output_dir}/{args.filename_tmpl.format(i)}'

        mmcv.imwrite(output_i, save_path_i)


if __name__ == '__main__':
    main()
