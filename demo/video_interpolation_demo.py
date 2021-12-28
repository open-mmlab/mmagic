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
        type=int,
        default=50,
        help='frames-per-second in the output video, which is needed when'
        'the input is image sequence and want to get an output video.')
    parser.add_argument(
        '--start_idx',
        type=int,
        default=0,
        help='index corresponds to the first frame of the sequence')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='batch size of video interpolation model')
    parser.add_argument(
        '--max_seq_len',
        type=int,
        default=None,
        help='maximum sequence length if recurrent framework is used')
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
                                                args.start_idx,
                                                args.max_seq_len,
                                                args.batch_size)

    fps = fps if fps > 0 else args.fps

    file_extension = os.path.splitext(args.output_dir)[1]
    if file_extension in VIDEO_EXTENSIONS:  # save as video
        h, w = output[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.output_dir, fourcc, fps, (w, h))
        for img in output:
            print(img.shape)
            video_writer.write(img)
        cv2.destroyAllWindows()
        video_writer.release()
    else:  # save as images
        for i, img in enumerate(output):
            save_path = f'{args.output_dir}/{args.filename_tmpl.format(i)}'
            mmcv.imwrite(img, save_path)


if __name__ == '__main__':
    main()
