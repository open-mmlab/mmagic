# Copyright (c) OpenMMLab. All rights reserved.
import os
import torch
import numpy as np
import mmcv
from typing import Dict, List
from torchvision import utils
from mmengine import mkdir_or_exist
from mmengine.dataset import Compose
from mmengine.dataset.utils import default_collate as collate

from mmedit.models.base_models import BaseTranslationModel
from mmedit.structures import EditDataSample
from .base_mmedit_inferencer import BaseMMEditInferencer, InputsType, PredType


class VideoRestorationInferencer(BaseMMEditInferencer):

    func_kwargs = dict(
        preprocess=['img'],
        forward=[],
        visualize=['img_out_dir'],
        postprocess=['print_result', 'pred_out_file', 'get_datasample'])

    def preprocess(self, img: InputsType) -> Dict:
        
        # build the data pipeline
        if self.model.cfg.get('demo_pipeline', None):
            test_pipeline = self.model.cfg.demo_pipeline
        elif self.model.cfg.get('test_pipeline', None):
            test_pipeline = self.model.cfg.test_pipeline
        else:
            test_pipeline = self.model.cfg.val_pipeline

        # check if the input is a video
        file_extension = osp.splitext(img_dir)[1]
        if file_extension in VIDEO_EXTENSIONS:
            video_reader = mmcv.VideoReader(img_dir)
            # load the images
            data = dict(img=[], img_path=None, key=img_dir)
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
            test_pipeline[0]['start_idx'] = start_idx
            test_pipeline[0]['filename_tmpl'] = filename_tmpl

            # prepare data
            sequence_length = len(glob.glob(osp.join(img_dir, '*')))
            lq_folder = osp.dirname(img_dir)
            key = osp.basename(img_dir)
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
            if window_size > 0:  # sliding window framework
                data = pad_sequence(data, window_size)
                result = []
                for i in range(0, data.size(1) - 2 * (window_size // 2)):
                    data_i = data[:, i:i + window_size].to(self.device)
                    result.append(self.model(inputs=data_i, mode='tensor').cpu())
                result = torch.stack(result, dim=1)
            else:  # recurrent framework
                if max_seq_len is None:
                    result = self.model(inputs=data.to(self.device), mode='tensor').cpu()
                else:
                    result = []
                    for i in range(0, data.size(1), max_seq_len):
                        result.append(
                            self.model(
                                inputs=data[:, i:i + max_seq_len].to(self.device),
                                mode='tensor').cpu())
                    result = torch.cat(result, dim=1)
        return result
    
    def visualize(self,
                preds: PredType,
                data: Dict = None,
                img_out_dir: str = '') -> List[np.ndarray]:
        
        results = (preds[:, [2, 1, 0]] + 1.) / 2.

        # save images
        mkdir_or_exist(os.path.dirname(img_out_dir))
        utils.save_image(results, img_out_dir)

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
