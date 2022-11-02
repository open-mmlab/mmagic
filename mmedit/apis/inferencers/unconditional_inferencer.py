# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from typing import Dict

from mmedit.structures import EditDataSample
from .base_mmedit_inferencer import BaseMMEditInferencer


class UnconditionalInferencer(BaseMMEditInferencer):

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
        result['text'] = data_sample.pred_text.item
        result['scores'] = float(np.mean(data_sample.pred_text.score))
        return result
