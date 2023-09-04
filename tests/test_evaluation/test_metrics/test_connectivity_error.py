# Copyright (c) OpenMMLab. All rights reserved.
import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from mmagic.datasets.transforms import LoadImageFromFile
from mmagic.evaluation.metrics import ConnectivityError


class TestMattingMetrics:

    @classmethod
    def setup_class(cls):
        # Make sure these values are immutable across different test cases.

        # This test depends on the interface of loading
        # if loading is changed, data should be change accordingly.
        test_path = Path(__file__).parent.parent.parent
        alpha_path = (
            test_path / 'data' / 'matting_dataset' / 'alpha' / 'GT05.jpg')

        results = dict(alpha_path=alpha_path)
        config = dict(key='alpha')
        image_loader = LoadImageFromFile(**config)
        results = image_loader(results)
        assert results['alpha'].ndim == 3

        gt_alpha = np.ones((32, 32), dtype=np.uint8) * 255
        trimap = np.zeros((32, 32), dtype=np.uint8)
        trimap[:16, :16] = 128
        trimap[16:, 16:] = 255
        # non-masked pred_alpha
        pred_alpha = torch.zeros((32, 32), dtype=torch.uint8)
        # masked pred_alpha
        masked_pred_alpha = pred_alpha.clone()
        masked_pred_alpha[trimap == 0] = 0
        masked_pred_alpha[trimap == 255] = 255

        gt_alpha = gt_alpha[None, ...]
        trimap = trimap[None, ...]

        cls.data_batch = [{
            'inputs': [],
            'data_samples': {
                'ori_trimap': torch.from_numpy(trimap),
                'ori_alpha': torch.from_numpy(gt_alpha),
            },
        }]

        cls.data_samples = [d_['data_samples'] for d_ in cls.data_batch]

        cls.bad_preds1_ = [{'pred_alpha': pred_alpha}]
        # pred_alpha should be masked by trimap before evaluation
        cls.bad_preds1 = copy.deepcopy(cls.data_samples)
        for d, p in zip(cls.bad_preds1, cls.bad_preds1_):
            d['output'] = p

        cls.bad_preds2_ = [{'pred_alpha': pred_alpha[0]}]
        # pred_alpha should be 3 dimensional
        cls.bad_preds2 = copy.deepcopy(cls.data_samples)
        for d, p in zip(cls.bad_preds2, cls.bad_preds2_):
            d['output'] = p

        cls.good_preds_ = [{'pred_alpha': masked_pred_alpha}]
        cls.good_preds = copy.deepcopy((cls.data_samples))
        for d, p in zip(cls.good_preds, cls.good_preds_):
            d['output'] = p

    def test_connectivity_error(self):
        """Test connectivity error for evaluating predicted alpha matte."""

        data_batch, bad_pred1, bad_pred2, good_pred = (
            self.data_batch,
            self.bad_preds1,
            self.bad_preds2,
            self.good_preds,
        )

        conn_err = ConnectivityError()

        with pytest.raises(ValueError):
            conn_err.process(data_batch, bad_pred1)

        with pytest.raises(ValueError):
            conn_err.process(data_batch, bad_pred2)

        # process 2 batches
        conn_err.process(data_batch, good_pred)
        conn_err.process(data_batch, good_pred)

        assert conn_err.results == [
            {
                'conn_err': 0.256,
            },
            {
                'conn_err': 0.256,
            },
        ]

        res = conn_err.compute_metrics(conn_err.results)

        assert list(res.keys()) == ['ConnectivityError']
        assert np.allclose(res['ConnectivityError'], 0.256)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
