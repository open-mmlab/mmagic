# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from pathlib import Path

import numpy as np
import pytest

from mmedit.datasets import AdobeComp1kDataset


class TestMattingDatasets:

    @classmethod
    def setup_class(cls):
        # create para for creating a dataset.
        cls.data_prefix = Path(__file__).parent.parent.parent / 'data'
        cls.ann_file = osp.join(cls.data_prefix, 'test_list.json')
        cls.pipeline = [
            dict(type='LoadImageFromFile', key='alpha', flag='grayscale')
        ]

    def test_comp1k_dataset(self):
        comp1k_dataset = AdobeComp1kDataset(self.ann_file, self.pipeline,
                                            self.data_prefix)
        first_data = comp1k_dataset[0]

        assert 'alpha' in first_data
        assert isinstance(first_data['alpha'], np.ndarray)
        assert first_data['alpha'].shape == (552, 800)

    def test_comp1k_evaluate(self):
        comp1k_dataset = AdobeComp1kDataset(self.ann_file, self.pipeline,
                                            self.data_prefix)

        with pytest.raises(TypeError):
            comp1k_dataset.evaluate('Not a list object')

        results = [{
            'pred_alpha': None,
            'eval_result': {
                'SAD': 26,
                'MSE': 0.006
            }
        }, {
            'pred_alpha': None,
            'eval_result': {
                'SAD': 24,
                'MSE': 0.004
            }
        }]

        eval_result = comp1k_dataset.evaluate(results)
        assert set(eval_result.keys()) == set(['SAD', 'MSE'])
        assert eval_result['SAD'] == 25
        assert eval_result['MSE'] == 0.005
