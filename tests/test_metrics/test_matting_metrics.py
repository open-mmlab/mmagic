# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmedit.metrics import SAD, ConnectivityError, GradientError, MattingMSE


class TestMattingMetrics:

    @classmethod
    def setup_class(cls):
        """Make sure these values are immutable across different test cases."""
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
        pred_alpha = pred_alpha.unsqueeze(0)
        masked_pred_alpha = masked_pred_alpha.unsqueeze(0)

        cls.data_batch = [{
            'inputs': [],
            'data_sample': {
                'ori_trimap': trimap,
                'ori_alpha': gt_alpha,
            },
        }]

        cls.bad_preds1 = [{'pred_alpha': dict(data=pred_alpha)}]
        # pred_alpha should be masked by trimap before evaluation

        cls.bad_preds2 = [{'pred_alpha': dict(data=pred_alpha[0])}]
        # pred_alpha should be 3 dimensional

        cls.good_preds = [{'pred_alpha': dict(data=masked_pred_alpha)}]

    def test_sad(self):
        """Test SAD for evaluating predicted alpha matte."""

        data_batch, bad_pred1, bad_pred2, good_pred = (
            self.data_batch,
            self.bad_preds1,
            self.bad_preds2,
            self.good_preds,
        )

        sad = SAD()

        with pytest.raises(ValueError):
            sad.process(data_batch, bad_pred1)

        with pytest.raises(ValueError):
            sad.process(data_batch, bad_pred2)

        # process 2 batches
        sad.process(data_batch, good_pred)
        sad.process(data_batch, good_pred)

        assert sad.results == [
            {
                'sad': 0.768,
            },
            {
                'sad': 0.768,
            },
        ]

        res = sad.compute_metrics(sad.results)

        assert list(res.keys()) == ['SAD']
        np.testing.assert_almost_equal(res['SAD'], 0.768)

    def test_mse(self):
        """Test MattingMSE for evaluating predicted alpha matte."""

        data_batch, bad_pred1, bad_pred2, good_pred = (
            self.data_batch,
            self.bad_preds1,
            self.bad_preds2,
            self.good_preds,
        )

        mse = MattingMSE()

        with pytest.raises(ValueError):
            mse.process(data_batch, bad_pred1)

        with pytest.raises(ValueError):
            mse.process(data_batch, bad_pred2)

        # process 2 batches
        mse.process(data_batch, good_pred)
        mse.process(data_batch, good_pred)

        assert mse.results == [
            {
                'mse': 3.0,
            },
            {
                'mse': 3.0,
            },
        ]

        res = mse.compute_metrics(mse.results)

        assert list(res.keys()) == ['MattingMSE']
        np.testing.assert_almost_equal(res['MattingMSE'], 3.0)

    def test_gradient_error(self):
        """Test gradient error for evaluating predicted alpha matte."""

        data_batch, bad_pred1, bad_pred2, good_pred = (
            self.data_batch,
            self.bad_preds1,
            self.bad_preds2,
            self.good_preds,
        )

        grad_err = GradientError()

        with pytest.raises(ValueError):
            grad_err.process(data_batch, bad_pred1)

        with pytest.raises(ValueError):
            grad_err.process(data_batch, bad_pred2)

        # process 2 batches
        grad_err.process(data_batch, good_pred)
        grad_err.process(data_batch, good_pred)

        assert len(grad_err.results) == 2
        for el in grad_err.results:
            assert list(el.keys()) == ['grad_err']
            np.testing.assert_almost_equal(el['grad_err'], 0.0028887)

        res = grad_err.compute_metrics(grad_err.results)

        assert list(res.keys()) == ['GradientError']
        np.testing.assert_almost_equal(el['grad_err'], 0.0028887)
        # assert np.allclose(res['GradientError'], 0.0028887)

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
