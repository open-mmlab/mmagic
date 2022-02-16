# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp

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


def video_interpolation_inference(model,
                                  input_dir,
                                  start_idx=0,
                                  end_idx=None,
                                  batch_size=4):
    """Inference image with the model.

    Args:
        model (nn.Module): The loaded model.
        input_dir (str): Directory of the input video.
        start_idx (int): The index corresponding to the first frame in the
            sequence. Default: 0.
        end_idx (int | None): The index corresponding to the last interpolated
            frame in the sequence. If it is None, interpolate to the last
            frame of video or sequence. Default: None.
        batch_size (int): Batch size. Default: 4.

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

    # check if the input is a video
    input_fps = 0.0
    file_extension = os.path.splitext(input_dir)[1]
    if file_extension in VIDEO_EXTENSIONS:
        video_reader = mmcv.VideoReader(input_dir)
        input_fps = video_reader.fps
        images = []
        # load the images
        for img in video_reader[start_idx:end_idx]:
            images.append(np.flip(img, axis=2))  # BGR --> RGB
    else:
        files = os.listdir(input_dir)
        files = [osp.join(input_dir, f) for f in files]
        files.sort()
        files = files[start_idx:end_idx]
        images = [read_image(f) for f in files]

    data = dict(inputs=images, inputs_path=None, key=input_dir)

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
    data = [test_pipeline(data)]
    data = collate(data, samples_per_gpu=1)['inputs']
    # data.shape: [1, t, c, h, w]

    # forward the model
    output_list = []
    data = model.split_frames(data)
    input_tensors = data.clone().detach()
    with torch.no_grad():
        length = data.shape[0]
        for start in range(0, length, batch_size):
            end = start + batch_size
            output = model(
                data[start:end].to(device), test_mode=True)['output']
            if len(output.shape) == 4:
                output = output.unsqueeze(1)
            output_list.append(output.cpu())

    output_tensors = torch.cat(output_list, dim=0)

    result = model.merge_frames(input_tensors, output_tensors)

    return result, input_fps
