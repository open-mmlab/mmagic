# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import cv2
import mmcv
import torch

from mmedit.apis import init_model, video_interpolation_inference

VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi')


def parse_args():
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
        '`fps_multiplier` is 0 and a video is desired as output.')
    parser.add_argument(
        '--fps_multiplier',
        type=float,
        default=0,
        help='multiply the fps based on the input video, if `fps_multiplier` '
        'is 0, `fps` will be utilized.')
    parser.add_argument(
        '--start_idx',
        type=int,
        default=0,
        help='the index of the first frame to be processed in the sequence')
    parser.add_argument(
        '--end_idx',
        type=int,
        default=None,
        help='The index corresponds to the last interpolated frame in the'
        'sequence. If it is None, interpolate to the last frame of video'
        'or sequence. Default: None.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='batch size of video interpolation model')
    parser.add_argument(
        '--filename_tmpl',
        default='{:08d}.png',
        help='template of the file names')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args


def main():
    """ Demo for video interpolation models.

    Note that we accept video as input(output), when 'input_dir'('output_dir')
    is set to the path to the video. But using videos introduces video
    compression, which lower the visual quality. If you want actual quality,
    please save them as separate images (.png).
    """

    args = parse_args()

    model = init_model(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))

    output, fps = video_interpolation_inference(model, args.input_dir,
                                                args.start_idx, args.end_idx,
                                                args.batch_size)

    if args.fps_multiplier:
        assert args.fps_multiplier > 0, '`fps_multiplier` cannot be negative'
        assert fps > 0, 'the input is not a video'
        fps = args.fps_multiplier * fps
    else:
        fps = args.fps if args.fps > 0 else fps

    file_extension = os.path.splitext(args.output_dir)[1]
    if file_extension in VIDEO_EXTENSIONS:  # save as video
        h, w = output[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.output_dir, fourcc, fps, (w, h))
        for img in output:
            video_writer.write(img)
        cv2.destroyAllWindows()
        video_writer.release()
    else:  # save as images
        for i, img in enumerate(output):
            save_path = f'{args.output_dir}/{args.filename_tmpl.format(i)}'
            mmcv.imwrite(img, save_path)


if __name__ == '__main__':
    main()
