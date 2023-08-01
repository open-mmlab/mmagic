# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os
import os.path as osp
from typing import Dict, List, Optional, Tuple, Union

import cv2
import mmcv
import mmengine
import numpy as np
import torch
from mmengine.dataset import Compose
from mmengine.logging import MMLogger
from mmengine.utils import ProgressBar

from mmagic.utils import tensor2img
from .base_mmagic_inferencer import (BaseMMagicInferencer, InputsType,
                                     PredType, ResType)
from .inference_functions import VIDEO_EXTENSIONS, pad_sequence


class VideoRestorationInferencer(BaseMMagicInferencer):
    """inferencer that predicts with video restoration models."""

    func_kwargs = dict(
        preprocess=['video'],
        forward=[],
        visualize=['result_out_dir'],
        postprocess=[])

    extra_parameters = dict(
        start_idx=0,
        filename_tmpl='{:08d}.png',
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
            # the first element in the pipeline must be
            # 'GenerateSegmentIndices'
            if test_pipeline[0]['type'] != 'GenerateSegmentIndices':
                raise TypeError('The first element in the pipeline must be '
                                f'"GenerateSegmentIndices", but got '
                                f'"{test_pipeline[0]["type"]}".')

            # specify start_idx and filename_tmpl
            test_pipeline[0]['start_idx'] = self.extra_parameters['start_idx']
            test_pipeline[0]['filename_tmpl'] = \
                self.extra_parameters['filename_tmpl']

            # prepare data
            sequence_length = len(glob.glob(osp.join(video, '*')))
            lq_folder = osp.dirname(video)
            key = osp.basename(video)
            data = dict(
                img_path=lq_folder,
                gt_path='',
                key=key,
                sequence_length=sequence_length)

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
        mmengine.utils.mkdir_or_exist(osp.dirname(result_out_dir))
        prog_bar = ProgressBar(preds.size(1))
        if file_extension in VIDEO_EXTENSIONS:  # save as video
            h, w = preds.shape[-2:]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(result_out_dir, fourcc, 25, (w, h))
            for i in range(0, preds.size(1)):
                img = tensor2img(preds[:, i, :, :, :])
                video_writer.write(img.astype(np.uint8))
                prog_bar.update()
            cv2.destroyAllWindows()
            video_writer.release()
        else:
            for i in range(self.extra_parameters['start_idx'],
                           self.extra_parameters['start_idx'] + preds.size(1)):
                output_i = \
                    preds[:, i - self.extra_parameters['start_idx'], :, :, :]
                output_i = tensor2img(output_i)
                filename_tmpl = self.extra_parameters['filename_tmpl']
                save_path_i = f'{result_out_dir}/{filename_tmpl.format(i)}'
                mmcv.imwrite(output_i, save_path_i)
                prog_bar.update()

        logger: MMLogger = MMLogger.get_current_instance()
        logger.info(f'Output video is save at {result_out_dir}.')

        return []

    def postprocess(
        self,
        preds: PredType,
        imgs: Optional[List[np.ndarray]] = None
    ) -> Union[ResType, Tuple[ResType, np.ndarray]]:
        """Postprocess is not needed in this inferencer."""
        logger: MMLogger = MMLogger.get_current_instance()
        logger.info('Postprocess is implemented in visualize process.')
        return None
