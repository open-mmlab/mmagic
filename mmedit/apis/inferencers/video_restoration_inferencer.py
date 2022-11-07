# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import torch
import numpy as np
import mmcv
from typing import Dict, List
from torchvision import utils
from mmengine import mkdir_or_exist
from mmengine.dataset import Compose

from mmedit.structures import EditDataSample
from .base_mmedit_inferencer import BaseMMEditInferencer, InputsType, PredType

VIDEO_EXTENSIONS = ('.mp4', '.mov')

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

    func_kwargs = dict(
        preprocess=['video'],
        forward=[],
        visualize=['result_out_dir'],
        postprocess=['print_result', 'pred_out_file', 'get_datasample'])

    def preprocess(self, video: InputsType) -> Dict:
        import pdb;pdb.set_trace();
        # hard code parameters for unused code
        infer_cfg = dict(
                    start_idx = 0,
                    filename_tmpl = '{08d}.png',
                    window_size = 0,
                    max_seq_len = None)
        self.start_idx = infer_cfg.start_idx
        self.filename_tmpl = infer_cfg.filename_tmpl
        self.window_size = infer_cfg.window_size
        self.max_seq_len = infer_cfg.max_seq_len
        
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
            # the first element in the pipeline must be 'GenerateSegmentIndices'
            if test_pipeline[0]['type'] != 'GenerateSegmentIndices':
                raise TypeError('The first element in the pipeline must be '
                                f'"GenerateSegmentIndices", but got '
                                f'"{test_pipeline[0]["type"]}".')

            # specify start_idx and filename_tmpl
            test_pipeline[0]['start_idx'] = self.start_idx
            test_pipeline[0]['filename_tmpl'] = self.filename_tmpl

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
        data = data['inputs'].unsqueeze(0) / 255.0  # in cpu

    def forward(self, inputs: InputsType) -> PredType:
        with torch.no_grad():
            if self.window_size > 0:  # sliding window framework
                data = pad_sequence(data, self.window_size)
                result = []
                for i in range(0, data.size(1) - 2 * (self.window_size // 2)):
                    data_i = data[:, i:i + self.window_size].to(self.device)
                    result.append(self.model(inputs=data_i, mode='tensor').cpu())
                result = torch.stack(result, dim=1)
            else:  # recurrent framework
                if self.max_seq_len is None:
                    result = self.model(inputs=data.to(self.device), mode='tensor').cpu()
                else:
                    result = []
                    for i in range(0, data.size(1), self.max_seq_len):
                        result.append(
                            self.model(
                                inputs=data[:, i:i + self.max_seq_len].to(self.device),
                                mode='tensor').cpu())
                    result = torch.cat(result, dim=1)
        return result
    
    def visualize(self,
                preds: PredType,
                data: Dict = None,
                result_out_dir: str = '') -> List[np.ndarray]:
        
        results = (preds[:, [2, 1, 0]] + 1.) / 2.

        # save images
        mkdir_or_exist(os.path.dirname(result_out_dir))
        utils.save_image(results, result_out_dir)

        return results

    def _pred2dict(self, data_sample: EditDataSample) -> Dict:
        """Extract elements necessary to represent a prediction into a
        dictionary. It's better to contain only basic data elements such as
        strings and numbers in order to guarantee it's json-serializable.

        Args:
            data_sample (EditDataSample): The data sample to be converted.

        Returns:
            dict: The output dictionary.
        """
        result = {}
        result['pred_alpha'] = data_sample.output.pred_alpha.data.cpu()
        return result
