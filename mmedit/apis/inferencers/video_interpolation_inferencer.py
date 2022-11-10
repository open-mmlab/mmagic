# Copyright (c) OpenMMLab. All rights reserved.
import math
import os
from typing import Dict, List, Optional, Tuple, Union

import cv2
import mmcv
import numpy as np
import torch
from mmengine.dataset import Compose
from mmengine.dataset.utils import default_collate as collate
from mmengine.fileio import FileClient
from mmengine.utils import ProgressBar

from .base_mmedit_inferencer import (BaseMMEditInferencer, InputsType,
                                     PredType, ResType)

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


class VideoInterpolationInferencer(BaseMMEditInferencer):
    """inferencer that predicts with video interpolation models."""

    func_kwargs = dict(
        preprocess=['video'],
        forward=['result_out_dir'],
        visualize=[],
        postprocess=[])

    def preprocess(self, video: InputsType) -> Dict:
        """Process the inputs into a model-feedable format.

        Args:
            video(InputsType): Video to be interpolated by models.

        Returns:
            video(InputsType): Video to be interpolated by models.
        """
        infer_cfg = dict(
            start_idx=0,
            end_idx=None,
            batch_size=4,
            fps_multiplier=0,
            fps=0,
            filename_tmpl='{08d}.png')
        self.start_idx = infer_cfg['start_idx']
        self.end_idx = infer_cfg['end_idx']
        self.batch_size = infer_cfg['batch_size']
        self.fps_multiplier = infer_cfg['fps_multiplier']
        self.fps = infer_cfg['fps']
        self.filename_tmpl = infer_cfg['filename_tmpl']

        # build the data pipeline
        if self.model.cfg.get('demo_pipeline', None):
            test_pipeline = self.model.cfg.demo_pipeline
        elif self.model.cfg.get('test_pipeline', None):
            test_pipeline = self.model.cfg.test_pipeline
        else:
            test_pipeline = self.model.cfg.val_pipeline

        # remove the data loading pipeline
        tmp_pipeline = []
        for pipeline in test_pipeline:
            if pipeline['type'] not in [
                    'GenerateSegmentIndices', 'LoadImageFromFile'
            ]:
                tmp_pipeline.append(pipeline)
        test_pipeline = tmp_pipeline

        # compose the pipeline
        self.test_pipeline = Compose(test_pipeline)

        return video

    def forward(self,
                inputs: InputsType,
                result_out_dir: InputsType = '') -> PredType:
        """Forward the inputs to the model.

        Args:
            inputs (InputsType): Input video directory.
            result_out_dir (str): Output directory of video.
                Defaults to ''.

        Returns:
            PredType: Result of forwarding
        """
        # check if the input is a video
        input_file_extension = os.path.splitext(inputs)[1]
        if input_file_extension in VIDEO_EXTENSIONS:
            source = mmcv.VideoReader(inputs)
            input_fps = source.fps
            length = source.frame_cnt
            from_video = True
            h, w = source.height, source.width
            if self.fps_multiplier:
                assert self.fps_multiplier > 0, \
                    '`fps_multiplier` cannot be negative'
                output_fps = self.fps_multiplier * input_fps
            else:
                output_fps = self.fps if self.fps > 0 else input_fps * 2
        else:
            raise ValueError('Input file is not a video, \
                which is not supported now.')

        # check if the output is a video
        output_file_extension = os.path.splitext(result_out_dir)[1]
        if output_file_extension in VIDEO_EXTENSIONS:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            target = cv2.VideoWriter(result_out_dir, fourcc, output_fps,
                                     (w, h))
            to_video = True
        else:
            to_video = False

        self.end_idx = min(self.end_idx,
                           length) if self.end_idx is not None else length

        # calculate step args
        step_size = self.model.step_frames * self.batch_size
        lenth_per_step = self.model.required_frames + \
            self.model.step_frames * (self.batch_size - 1)
        repeat_frame = self.model.required_frames - self.model.step_frames

        prog_bar = ProgressBar(
            math.ceil(
                (self.end_idx + step_size - lenth_per_step - self.start_idx) /
                step_size))
        for self.start_index in range(self.start_idx, self.end_idx, step_size):
            images = read_frames(
                source,
                self.start_index,
                lenth_per_step,
                from_video,
                end_index=self.end_idx)

            # data prepare
            data = dict(img=images, inputs_path=None, key=inputs)
            data = self.test_pipeline(data)['inputs'] / 255.0
            data = collate([data])
            # data.shape: [1, t, c, h, w]

            # forward the model
            data = self.model.split_frames(data)
            input_tensors = data.clone().detach()
            with torch.no_grad():
                output = self.model(data.to(self.device), mode='tensor')
                if len(output.shape) == 4:
                    output = output.unsqueeze(1)
                output_tensors = output.cpu()
                if len(output_tensors.shape) == 4:
                    output_tensors = output_tensors.unsqueeze(1)
                result = self.model.merge_frames(input_tensors, output_tensors)
            if not self.start_idx == self.start_index:
                result = result[repeat_frame:]
            prog_bar.update()

            # save frames
            if to_video:
                for frame in result:
                    target.write(frame)
            else:
                raise ValueError('Output file is not a video, \
                    which is not supported now.')

            if self.start_index + lenth_per_step >= self.end_idx:
                break

        print()
        print(f'Output dir: {result_out_dir}')
        if to_video:
            target.release()

        return {}

    def visualize(self,
                  preds: PredType,
                  result_out_dir: str = '') -> List[np.ndarray]:
        """Visualize is not needed in this inferencer."""
        pass

    def postprocess(
        self,
        preds: PredType,
        imgs: Optional[List[np.ndarray]] = None
    ) -> Union[ResType, Tuple[ResType, np.ndarray]]:
        """Postprocess is not needed in this inferencer."""
        pass
