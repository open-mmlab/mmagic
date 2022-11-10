# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from typing import Dict, List, Optional, Tuple, Union

import cv2
import mmcv
import numpy as np
import torch
from mmengine.dataset import Compose

from mmedit.utils import tensor2img
from .base_mmedit_inferencer import (BaseMMEditInferencer, InputsType,
                                     PredType, ResType)

VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi')


def pad_sequence(data, window_size):
    """Pad frame sequence data.

    Args:
        data (Tensor): The frame sequence data.
        window_size (int): The window size used in sliding-window framework.

    Returns:
        data (Tensor): The padded result.
    """

    padding = window_size // 2

    data = torch.cat([
        data[:, 1 + padding:1 + 2 * padding].flip(1), data,
        data[:, -1 - 2 * padding:-1 - padding].flip(1)
    ],
                     dim=1)

    return data


class VideoRestorationInferencer(BaseMMEditInferencer):
    """inferencer that predicts with video restoration models."""

    func_kwargs = dict(
        preprocess=['video'],
        forward=[],
        visualize=['result_out_dir'],
        postprocess=[])

    extra_parameters = dict(
        start_idx=0,
        filename_tmpl='{08d}.png',
        window_size=0,
        max_seq_len=None)

    def preprocess(self, video: InputsType) -> Dict:
        """Process the inputs into a model-feedable format.

        Args:
            video(InputsType): Video to be restored by models.

        Returns:
            results(InputsType): Results of preprocess.
        """
        # build the data pipeline
        if self.model.cfg.get('demo_pipeline', None):
            test_pipeline = self.model.cfg.demo_pipeline
        elif self.model.cfg.get('test_pipeline', None):
            test_pipeline = self.model.cfg.test_pipeline
        else:
            test_pipeline = self.model.cfg.val_pipeline

        # check if the input is a video
        file_extension = osp.splitext(video)[1]
        if file_extension in VIDEO_EXTENSIONS:
            video_reader = mmcv.VideoReader(video)
            # load the images
            data = dict(img=[], img_path=None, key=video)
            for frame in video_reader:
                data['img'].append(np.flip(frame, axis=2))

            # remove the data loading pipeline
            tmp_pipeline = []
            for pipeline in test_pipeline:
                if pipeline['type'] not in [
                        'GenerateSegmentIndices', 'LoadImageFromFile'
                ]:
                    tmp_pipeline.append(pipeline)
            test_pipeline = tmp_pipeline
        else:
            raise ValueError('Input file is not a video, \
                which is not supported now.')

        # compose the pipeline
        test_pipeline = Compose(test_pipeline)
        data = test_pipeline(data)
        results = data['inputs'].unsqueeze(0) / 255.0  # in cpu

        return results

    def forward(self, inputs: InputsType) -> PredType:
        """Forward the inputs to the model.

        Args:
            inputs (InputsType): Images array of input video.

        Returns:
            PredType: Results of forwarding
        """
        with torch.no_grad():
            if self.extra_parameters[
                    'window_size'] > 0:  # sliding window framework
                data = pad_sequence(inputs,
                                    self.extra_parameters['window_size'])
                result = []
                # yapf: disable
                for i in range(0, data.size(1) - 2 * (self.extra_parameters['window_size'] // 2)):  # noqa
                    # yapf: enable
                    data_i = data[:, i:i +
                                  self.extra_parameters['window_size']].to(
                                      self.device)
                    result.append(
                        self.model(inputs=data_i, mode='tensor').cpu())
                result = torch.stack(result, dim=1)
            else:  # recurrent framework
                if self.extra_parameters['max_seq_len'] is None:
                    result = self.model(
                        inputs=inputs.to(self.device), mode='tensor').cpu()
                else:
                    result = []
                    for i in range(0, inputs.size(1),
                                   self.extra_parameters['max_seq_len']):
                        result.append(
                            self.model(
                                inputs=inputs[:, i:i + self.
                                              extra_parameters['max_seq_len']].
                                to(self.device),
                                mode='tensor').cpu())
                    result = torch.cat(result, dim=1)
        return result

    def visualize(self,
                  preds: PredType,
                  result_out_dir: str = '') -> List[np.ndarray]:
        """Visualize predictions.

        Args:
            preds (List[Union[str, np.ndarray]]): Forward results
                by the inferencer.
            data (List[Dict]): Not needed by this kind of inferencer.
            result_out_dir (str): Output directory of image.
                Defaults to ''.

        Returns:
            List[np.ndarray]: Result of visualize
        """
        file_extension = os.path.splitext(result_out_dir)[1]
        if file_extension in VIDEO_EXTENSIONS:  # save as video
            h, w = preds.shape[-2:]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(result_out_dir, fourcc, 25, (w, h))
            for i in range(0, preds.size(1)):
                img = tensor2img(preds[:, i, :, :, :])
                video_writer.write(img.astype(np.uint8))
            cv2.destroyAllWindows()
            video_writer.release()
        else:
            raise ValueError('Output file is not a video, \
                which is not supported now.')

        return []

    def postprocess(
        self,
        preds: PredType,
        imgs: Optional[List[np.ndarray]] = None
    ) -> Union[ResType, Tuple[ResType, np.ndarray]]:
        """Postprocess is not needed in this inferencer."""
        pass
