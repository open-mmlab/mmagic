# Copyright (c) OpenMMLab. All rights reserved.
import math
import os
import os.path as osp
from typing import Dict, List, Optional, Tuple, Union

import cv2
import mmcv
import numpy as np
import torch
from mmengine.dataset import Compose
from mmengine.dataset.utils import default_collate as collate
from mmengine.logging import MMLogger
from mmengine.utils import ProgressBar

from .base_mmedit_inferencer import (BaseMMEditInferencer, InputsType,
                                     PredType, ResType)
from .inference_functions import VIDEO_EXTENSIONS, read_frames, read_image


class VideoInterpolationInferencer(BaseMMEditInferencer):
    """inferencer that predicts with video interpolation models."""

    func_kwargs = dict(
        preprocess=['video'],
        forward=['result_out_dir'],
        visualize=[],
        postprocess=[])

    extra_parameters = dict(
        start_idx=0,
        end_idx=None,
        batch_size=4,
        fps_multiplier=0,
        fps=0,
        filename_tmpl='{:08d}.png')

    def preprocess(self, video: InputsType) -> Dict:
        """Process the inputs into a model-feedable format.

        Args:
            video(InputsType): Video to be interpolated by models.

        Returns:
            video(InputsType): Video to be interpolated by models.
        """
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
            if self.extra_parameters['fps_multiplier']:
                assert self.extra_parameters['fps_multiplier'] > 0, \
                    '`fps_multiplier` cannot be negative'
                output_fps = \
                    self.extra_parameters['fps_multiplier'] * input_fps
            else:
                fps = self.extra_parameters['fps']
                output_fps = fps if fps > 0 else input_fps * 2
        else:
            files = os.listdir(inputs)
            files = [osp.join(inputs, f) for f in files]
            files.sort()
            source = files
            length = files.__len__()
            from_video = False
            example_frame = read_image(files[0])
            h, w = example_frame.shape[:2]
            fps = self.extra_parameters['fps']
            output_fps = fps if fps > 0 else 60

        # check if the output is a video
        output_file_extension = os.path.splitext(result_out_dir)[1]
        if output_file_extension in VIDEO_EXTENSIONS:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            target = cv2.VideoWriter(result_out_dir, fourcc, output_fps,
                                     (w, h))
            to_video = True
        else:
            to_video = False

        self.extra_parameters['end_idx'] = min(
            self.extra_parameters['end_idx'], length) \
            if self.extra_parameters['end_idx'] is not None else length

        # calculate step args
        step_size = \
            self.model.step_frames * self.extra_parameters['batch_size']
        lenth_per_step = self.model.required_frames + \
            self.model.step_frames * (self.extra_parameters['batch_size'] - 1)
        repeat_frame = self.model.required_frames - self.model.step_frames

        prog_bar = ProgressBar(
            math.ceil((self.extra_parameters['end_idx'] + step_size -
                       lenth_per_step - self.extra_parameters['start_idx']) /
                      step_size))
        output_index = self.extra_parameters['start_idx']
        for start_index in range(self.extra_parameters['start_idx'],
                                 self.extra_parameters['end_idx'], step_size):
            images = read_frames(
                source,
                start_index,
                lenth_per_step,
                from_video,
                end_index=self.extra_parameters['end_idx'])

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
            if not self.extra_parameters['start_idx'] == start_index:
                result = result[repeat_frame:]
            prog_bar.update()

            # save frames
            if to_video:
                for frame in result:
                    target.write(frame)
            else:
                filename_tmpl = self.extra_parameters['filename_tmpl']
                for frame in result:
                    save_path = osp.join(result_out_dir,
                                         filename_tmpl.format(output_index))
                    mmcv.imwrite(frame, save_path)
                    output_index += 1

            if start_index + lenth_per_step >= \
               self.extra_parameters['end_idx']:
                break

        logger: MMLogger = MMLogger.get_current_instance()
        logger.info(f'Output video is save at {result_out_dir}.')
        if to_video:
            target.release()

        return {}

    def visualize(self,
                  preds: PredType,
                  result_out_dir: str = '') -> List[np.ndarray]:
        """Visualize is not needed in this inferencer."""
        logger: MMLogger = MMLogger.get_current_instance()
        logger.info('Visualization is implemented in forward process.')
        return None

    def postprocess(
        self,
        preds: PredType,
        imgs: Optional[List[np.ndarray]] = None
    ) -> Union[ResType, Tuple[ResType, np.ndarray]]:
        """Postprocess is not needed in this inferencer."""
        logger: MMLogger = MMLogger.get_current_instance()
        logger.info('Postprocess is implemented in forward process.')
        return None
