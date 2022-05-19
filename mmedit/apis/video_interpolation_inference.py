# Copyright (c) OpenMMLab. All rights reserved.
import math
import os
import os.path as osp

import cv2
import mmcv
import numpy as np
import torch
from mmcv.fileio import FileClient
from mmcv.parallel import collate

from mmedit.datasets.pipelines import Compose

VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi')
FILE_CLIENT = FileClient('disk')


def read_image(filepath):
    """Read image from file.

    Args:
        filepath (str): File path.

    Returns:
        image (np.array): Image.
    """
    img_bytes = FILE_CLIENT.get(filepath)
    image = mmcv.imfrombytes(
        img_bytes, flag='color', channel_order='rgb', backend='pillow')
    return image


def read_frames(source, start_index, num_frames, from_video, end_index):
    """Read frames from file or video.

    Args:
        source (list | mmcv.VideoReader): Source of frames.
        start_index (int): Start index of frames.
        num_frames (int): frames number to be read.
        from_video (bool): Weather read frames from video.
        end_index (int): The end index of frames.

    Returns:
        images (np.array): Images.
    """
    images = []
    last_index = min(start_index + num_frames, end_index)
    # read frames from video
    if from_video:
        for index in range(start_index, last_index):
            if index >= source.frame_cnt:
                break
            images.append(np.flip(source.get_frame(index), axis=2))
    else:
        files = source[start_index:last_index]
        images = [read_image(f) for f in files]
    return images


def video_interpolation_inference(model,
                                  input_dir,
                                  output_dir,
                                  start_idx=0,
                                  end_idx=None,
                                  batch_size=4,
                                  fps_multiplier=0,
                                  fps=0,
                                  filename_tmpl='{:08d}.png'):
    """Inference image with the model.

    Args:
        model (nn.Module): The loaded model.
        input_dir (str): Directory of the input video.
        output_dir (str): Directory of the output video.
        start_idx (int): The index corresponding to the first frame in the
            sequence. Default: 0
        end_idx (int | None): The index corresponding to the last interpolated
            frame in the sequence. If it is None, interpolate to the last
            frame of video or sequence. Default: None
        batch_size (int): Batch size. Default: 4
        fps_multiplier (float): multiply the fps based on the input video.
            Default: 0.
        fps (float): frame rate of the output video. Default: 0.
        filename_tmpl (str): template of the file names. Default: '{:08d}.png'

    Returns:
        output (list[numpy.array]): The predicted interpolation result.
            It is an image sequence.
        input_fps (float): The fps of input video. If the input is an image
            sequence, input_fps=0.0
    """

    device = next(model.parameters()).device  # model device

    # build the data pipeline
    if model.cfg.get('demo_pipeline', None):
        test_pipeline = model.cfg.demo_pipeline
    elif model.cfg.get('test_pipeline', None):
        test_pipeline = model.cfg.test_pipeline
    else:
        test_pipeline = model.cfg.val_pipeline

    # remove the data loading pipeline
    tmp_pipeline = []
    for pipeline in test_pipeline:
        if pipeline['type'] not in [
                'GenerateSegmentIndices', 'LoadImageFromFileList',
                'LoadImageFromFile'
        ]:
            tmp_pipeline.append(pipeline)
    test_pipeline = tmp_pipeline

    # compose the pipeline
    test_pipeline = Compose(test_pipeline)

    # check if the input is a video
    input_file_extension = os.path.splitext(input_dir)[1]
    if input_file_extension in VIDEO_EXTENSIONS:
        source = mmcv.VideoReader(input_dir)
        input_fps = source.fps
        length = source.frame_cnt
        from_video = True
        h, w = source.height, source.width
        if fps_multiplier:
            assert fps_multiplier > 0, '`fps_multiplier` cannot be negative'
            output_fps = fps_multiplier * input_fps
        else:
            output_fps = fps if fps > 0 else input_fps * 2
    else:
        files = os.listdir(input_dir)
        files = [osp.join(input_dir, f) for f in files]
        files.sort()
        source = files
        length = files.__len__()
        from_video = False
        example_frame = read_image(files[0])
        h, w = example_frame.shape[:2]
        output_fps = fps

    # check if the output is a video
    output_file_extension = os.path.splitext(output_dir)[1]
    if output_file_extension in VIDEO_EXTENSIONS:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        target = cv2.VideoWriter(output_dir, fourcc, output_fps, (w, h))
        to_video = True
    else:
        to_video = False

    end_idx = min(end_idx, length) if end_idx is not None else length

    # calculate step args
    step_size = model.step_frames * batch_size
    lenth_per_step = model.required_frames + model.step_frames * (
        batch_size - 1)
    repeat_frame = model.required_frames - model.step_frames

    prog_bar = mmcv.ProgressBar(
        math.ceil(
            (end_idx + step_size - lenth_per_step - start_idx) / step_size))
    output_index = start_idx
    for start_index in range(start_idx, end_idx, step_size):
        images = read_frames(
            source, start_index, lenth_per_step, from_video, end_index=end_idx)

        # data prepare
        data = dict(inputs=images, inputs_path=None, key=input_dir)
        data = [test_pipeline(data)]
        data = collate(data, samples_per_gpu=1)['inputs']
        # data.shape: [1, t, c, h, w]

        # forward the model
        data = model.split_frames(data)
        input_tensors = data.clone().detach()
        with torch.no_grad():
            output = model(data.to(device), test_mode=True)['output']
            if len(output.shape) == 4:
                output = output.unsqueeze(1)
            output_tensors = output.cpu()
            if len(output_tensors.shape) == 4:
                output_tensors = output_tensors.unsqueeze(1)
            result = model.merge_frames(input_tensors, output_tensors)
        if not start_idx == start_index:
            result = result[repeat_frame:]
        prog_bar.update()

        # save frames
        if to_video:
            for frame in result:
                target.write(frame)
        else:
            for frame in result:
                save_path = osp.join(output_dir,
                                     filename_tmpl.format(output_index))
                mmcv.imwrite(frame, save_path)
                output_index += 1

        if start_index + lenth_per_step >= end_idx:
            break

    print()
    print(f'Output dir: {output_dir}')
    if to_video:
        target.release()
