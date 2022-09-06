# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch

from mmedit.apis import init_model, video_interpolation_inference
from mmedit.utils import modify_args

VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi')


def parse_args():
    modify_args()
    parser = argparse.ArgumentParser(description='Restoration demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('input_dir', help='directory of the input video')
    parser.add_argument('output_dir', help='directory of the output video')
    parser.add_argument(
        '--fps',
        type=float,
        default=0,
        help='frame rate of the output video, which is needed when '
        '`fps-multiplier` is 0 and a video is desired as output.')
    parser.add_argument(
        '--fps-multiplier',
        type=float,
        default=0,
        help='multiply the fps based on the input video, if `fps-multiplier` '
        'is 0, `fps` will be utilized.')
    parser.add_argument(
        '--start-idx',
        type=int,
        default=0,
        help='the index of the first frame to be processed in the sequence')
    parser.add_argument(
        '--end-idx',
        type=int,
        default=None,
        help='The index corresponds to the last interpolated frame in the'
        'sequence. If it is None, interpolate to the last frame of video'
        'or sequence. Default: None.')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='batch size of video interpolation model')
    parser.add_argument(
        '--filename-tmpl',
        default='{:08d}.png',
        help='template of the file names')
    parser.add_argument(
        '--device', type=int, default=None, help='CUDA device id')
    args = parser.parse_args()
    return args


def main():
    """Demo for video interpolation models.

    Note that we accept video as input(output), when 'input_dir'('output_dir')
    is set to the path to the video. But using videos introduces video
    compression, which lower the visual quality. If you want actual quality,
    please save them as separate images (.png).
    """

    args = parse_args()

    if args.device < 0 or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', args.device)

    model = init_model(args.config, args.checkpoint, device=device)

    video_interpolation_inference(
        model=model,
        input_dir=args.input_dir,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        batch_size=args.batch_size,
        fps_multiplier=args.fps_multiplier,
        fps=args.fps,
        output_dir=args.output_dir,
        filename_tmpl=args.filename_tmpl)


if __name__ == '__main__':
    main()
