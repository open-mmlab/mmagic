# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os.path as osp

import mmcv
import numpy as np
import torch

from mmedit.datasets.pipelines import Compose

VIDEO_EXTENSIONS = ('.mp4', '.mov')


def pad_sequence(data, window_size):
    padding = window_size // 2

    data = torch.cat([
        data[:, 1 + padding:1 + 2 * padding].flip(1), data,
        data[:, -1 - 2 * padding:-1 - padding].flip(1)
    ],
                     dim=1)

    return data


def restoration_video_inference(model,
                                img_dir,
                                window_size,
                                start_idx,
                                filename_tmpl,
                                max_seq_len=None):
    """Inference image with the model.

    Args:
        model (nn.Module): The loaded model.
        img_dir (str): Directory of the input video.
        window_size (int): The window size used in sliding-window framework.
            This value should be set according to the settings of the network.
            A value smaller than 0 means using recurrent framework.
        start_idx (int): The index corresponds to the first frame in the
            sequence.
        filename_tmpl (str): Template for file name.
        max_seq_len (int | None): The maximum sequence length that the model
            processes. If the sequence length is larger than this number,
            the sequence is split into multiple segments. If it is None,
            the entire sequence is processed at once.

    Returns:
        Tensor: The predicted restoration result.
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
    file_extension = osp.splitext(img_dir)[1]
    if file_extension in VIDEO_EXTENSIONS:
        video_reader = mmcv.VideoReader(img_dir)
        # load the images
        data = dict(lq=[], lq_path=None, key=img_dir)
        for frame in video_reader:
            data['lq'].append(np.flip(frame, axis=2))

        # remove the data loading pipeline
        tmp_pipeline = []
        for pipeline in test_pipeline:
            if pipeline['type'] not in [
                    'GenerateSegmentIndices', 'LoadImageFromFileList'
            ]:
                tmp_pipeline.append(pipeline)
        test_pipeline = tmp_pipeline
    else:
        # the first element in the pipeline must be 'GenerateSegmentIndices'
        if test_pipeline[0]['type'] != 'GenerateSegmentIndices':
            raise TypeError('The first element in the pipeline must be '
                            f'"GenerateSegmentIndices", but got '
                            f'"{test_pipeline[0]["type"]}".')

        # specify start_idx and filename_tmpl
        test_pipeline[0]['start_idx'] = start_idx
        test_pipeline[0]['filename_tmpl'] = filename_tmpl

        # prepare data
        sequence_length = len(glob.glob(osp.join(img_dir, '*')))
        lq_folder = osp.dirname(img_dir)
        key = osp.basename(img_dir)
        data = dict(
            lq_path=lq_folder,
            gt_path='',
            key=key,
            sequence_length=sequence_length)

    # compose the pipeline
    test_pipeline = Compose(test_pipeline)
    data = test_pipeline(data)
    data = data['lq'].unsqueeze(0)  # in cpu

    # forward the model
    with torch.no_grad():
        if window_size > 0:  # sliding window framework
            data = pad_sequence(data, window_size)
            result = []
            for i in range(0, data.size(1) - 2 * (window_size // 2)):
                data_i = data[:, i:i + window_size].to(device)
                result.append(model(lq=data_i, test_mode=True)['output'].cpu())
            result = torch.stack(result, dim=1)
        else:  # recurrent framework
            if max_seq_len is None:
                result = model(
                    lq=data.to(device), test_mode=True)['output'].cpu()
            else:
                result = []
                for i in range(0, data.size(1), max_seq_len):
                    result.append(
                        model(
                            lq=data[:, i:i + max_seq_len].to(device),
                            test_mode=True)['output'].cpu())
                result = torch.cat(result, dim=1)
    return result
