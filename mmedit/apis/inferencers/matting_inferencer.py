# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Dict, List

import mmcv
import numpy as np
import torch
from mmengine import mkdir_or_exist
from mmengine.dataset import Compose
from mmengine.dataset.utils import default_collate as collate
from torch.nn.parallel import scatter

from mmedit.structures import EditDataSample
from .base_mmedit_inferencer import BaseMMEditInferencer, InputsType, PredType


class MattingInferencer(BaseMMEditInferencer):
    """inferencer that predicts with matting models."""

    func_kwargs = dict(
        preprocess=['img', 'trimap'],
        forward=[],
        visualize=['result_out_dir'],
        postprocess=[])

    def preprocess(self, img: InputsType, trimap: InputsType) -> Dict:
        """Process the inputs into a model-feedable format.

        Args:
            img(InputsType): Image to be processed by models.
            mask(InputsType): Mask corresponding to the input image.

        Returns:
            results(Dict): Results of preprocess.
        """
        # remove alpha from test_pipeline
        keys_to_remove = ['alpha', 'ori_alpha']
        for key in keys_to_remove:
            for pipeline in list(self.cfg.test_pipeline):
                if 'key' in pipeline and key == pipeline['key']:
                    self.cfg.test_pipeline.remove(pipeline)
                if 'keys' in pipeline and key in pipeline['keys']:
                    pipeline['keys'].remove(key)
                    if len(pipeline['keys']) == 0:
                        self.cfg.test_pipeline.remove(pipeline)
                if 'meta_keys' in pipeline and key in pipeline['meta_keys']:
                    pipeline['meta_keys'].remove(key)

        # build the data pipeline
        test_pipeline = Compose(self.cfg.test_pipeline)
        # prepare data
        data = dict(merged_path=img, trimap_path=trimap)
        _data = test_pipeline(data)
        trimap = _data['data_samples'].trimap.data
        preprocess_res = dict()
        preprocess_res['inputs'] = torch.cat([_data['inputs'], trimap],
                                             dim=0).float()
        preprocess_res = collate([preprocess_res])
        preprocess_res['data_samples'] = [_data['data_samples']]
        preprocess_res['mode'] = 'predict'
        if 'cuda' in str(self.device):
            preprocess_res = scatter(preprocess_res, [self.device])[0]

        return preprocess_res

    def forward(self, inputs: InputsType) -> PredType:
        """Forward the inputs to the model."""
        with torch.no_grad():
            return self.model(**inputs)

    def visualize(self,
                  preds: PredType,
                  result_out_dir: str = None) -> List[np.ndarray]:
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
        result = preds[0].output
        result = result.pred_alpha.data.cpu()

        # save images
        if result_out_dir:
            mkdir_or_exist(os.path.dirname(result_out_dir))
            mmcv.imwrite(result.numpy(), result_out_dir)

        return result

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
