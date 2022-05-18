# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest

from mmedit.metrics import MSE, SAD, ConnectivityError, GradientError


class TestMattingMetrics:

    @classmethod
    def setup_class(cls):
        'Make sure these values are immutable across different test cases.'
        gt_alpha = np.ones((2, 32, 32)) * 255
        trimap = np.zeros((2, 32, 32), dtype=np.uint8)
        trimap[:, :16, :16] = 128
        trimap[:, 16:, 16:] = 255
        # non-masked pred_alpha
        pred_alpha = np.zeros((2, 32, 32))
        # masked pred_alpha
        masked_pred_alpha = pred_alpha.copy()
        masked_pred_alpha[trimap == 0] = 0
        masked_pred_alpha[trimap == 255] = 255

        cls.data = (trimap, gt_alpha, pred_alpha, masked_pred_alpha)

    def test_sad(self):
        """Test SAD for evaluating predicted alpha matte."""

        trimap, gt_alpha, pred_alpha, masked_pred_alpha = self.data
        sad = SAD()

        data_batch = {
            'img': [],
            'data_sample': {
                'trimap': trimap,
                'gt_alpha': gt_alpha,
            },
        }
        predictions = {
            'img': [],
            'data_sample': {
                'pred_alpha': pred_alpha
            },  # test bad case first
        }

        with pytest.raises(ValueError):
            # pred_alpha should be masked by trimap before evaluation
            sad.process(data_batch, predictions)

        with pytest.raises(ValueError):
            # input should all be three dimensional
            predictions['data_sample']['pred_alpha'] = pred_alpha[0]
            sad.process(data_batch, predictions)

        # ! alpha should be masked by trimap before evaluation !
        predictions['data_sample']['pred_alpha'] = masked_pred_alpha

        # Another batch consists of only 1 sample
        data_batch1 = {
            'img': [],
            'data_sample': {
                'trimap': trimap[0:1],
                'gt_alpha': gt_alpha[0:1]
            },
        }
        predictions1 = {
            'img': [],
            'data_sample': {
                'pred_alpha': masked_pred_alpha[0:1]
            },
        }

        # process 2 batches
        sad.process(data_batch, predictions)
        sad.process(data_batch1, predictions1)

        assert sad.results == [
            {
                'sad_sum': 1.536,
                'n': 2
            },
            {
                'sad_sum': 0.768,
                'n': 1
            },
        ]

        res = sad.compute_metrics(sad.results)

        assert list(res.keys()) == ['SAD']
        np.testing.assert_almost_equal(res['SAD'], 0.768)

    def test_mse(self):
        """Test MSE for evaluating predicted alpha matte."""

        trimap, gt_alpha, pred_alpha, masked_pred_alpha = self.data
        mse = MSE()

        data_batch = {
            'img': [],
            'data_sample': {
                'trimap': trimap,
                'gt_alpha': gt_alpha,
            },
        }
        predictions = {
            'img': [],
            'data_sample': {
                'pred_alpha': pred_alpha
            },  # test bad case first
        }

        with pytest.raises(ValueError):
            # pred_alpha should be masked by trimap before evaluation
            mse.process(data_batch, predictions)

        with pytest.raises(ValueError):
            # input should all be three dimensional
            predictions['data_sample']['pred_alpha'] = pred_alpha[0]
            mse.process(data_batch, predictions)

        # ! alpha should be masked by trimap before evaluation !
        predictions['data_sample']['pred_alpha'] = masked_pred_alpha

        # Another batch consists of only 1 sample
        data_batch1 = {
            'img': [],
            'data_sample': {
                'trimap': trimap[0:1],
                'gt_alpha': gt_alpha[0:1]
            },
        }
        predictions1 = {
            'img': [],
            'data_sample': {
                'pred_alpha': masked_pred_alpha[0:1]
            },
        }

        # process 2 batches
        mse.process(data_batch, predictions)
        mse.process(data_batch1, predictions1)

        assert mse.results == [
            {
                'mse': 3.0,
                # 'n': 1
            },
            {
                'mse': 3.0,
                # 'n': 1
            },
            {
                'mse': 3.0,
                # 'n': 1
            },
        ]

        res = mse.compute_metrics(mse.results)

        assert list(res.keys()) == ['MSE']
        np.testing.assert_almost_equal(res['MSE'], 3.0)

    def test_gradient_error(self):
        """Test gradient error for evaluating predicted alpha matte."""

        trimap, gt_alpha, pred_alpha, masked_pred_alpha = self.data
        grad_err = GradientError()

        data_batch = {
            'img': [],
            'data_sample': {
                'trimap': trimap,
                'gt_alpha': gt_alpha,
            },
        }
        predictions = {
            'img': [],
            'data_sample': {
                'pred_alpha': pred_alpha
            },  # test bad case first
        }

        with pytest.raises(ValueError):
            # pred_alpha should be masked by trimap before evaluation
            grad_err.process(data_batch, predictions)

        with pytest.raises(ValueError):
            # input should all be three dimensional
            predictions['data_sample']['pred_alpha'] = pred_alpha[0]
            grad_err.process(data_batch, predictions)

        # ! alpha should be masked by trimap before evaluation !
        predictions['data_sample']['pred_alpha'] = masked_pred_alpha

        # Another batch consists of only 1 sample
        data_batch1 = {
            'img': [],
            'data_sample': {
                'trimap': trimap[0:1],
                'gt_alpha': gt_alpha[0:1]
            },
        }
        predictions1 = {
            'img': [],
            'data_sample': {
                'pred_alpha': masked_pred_alpha[0:1]
            },
        }

        # process 2 batches
        grad_err.process(data_batch, predictions)
        grad_err.process(data_batch1, predictions1)

        assert len(grad_err.results) == 3
        for el in grad_err.results:
            assert list(el.keys()) == ['grad_err']
            np.testing.assert_almost_equal(el['grad_err'], 0.0028887)

        res = grad_err.compute_metrics(grad_err.results)

        assert list(res.keys()) == ['GradientError']
        np.testing.assert_almost_equal(el['grad_err'], 0.0028887)
        # assert np.allclose(res['GradientError'], 0.0028887)

    def test_connectivity_error(self):
        """Test connectivity error for evaluating predicted alpha matte."""

        trimap, gt_alpha, pred_alpha, masked_pred_alpha = self.data
        conn_err = ConnectivityError()

        data_batch = {
            'img': [],
            'data_sample': {
                'trimap': trimap,
                'gt_alpha': gt_alpha,
            },
        }
        predictions = {
            'img': [],
            'data_sample': {
                'pred_alpha': pred_alpha
            },  # test bad case first
        }

        with pytest.raises(ValueError):
            # pred_alpha should be masked by trimap before evaluation
            conn_err.process(data_batch, predictions)

        with pytest.raises(ValueError):
            # input should all be three dimensional
            predictions['data_sample']['pred_alpha'] = pred_alpha[0]
            conn_err.process(data_batch, predictions)

        # ! alpha should be masked by trimap before evaluation !
        predictions['data_sample']['pred_alpha'] = masked_pred_alpha

        # Another batch consists of only 1 sample
        data_batch1 = {
            'img': [],
            'data_sample': {
                'trimap': trimap[0:1],
                'gt_alpha': gt_alpha[0:1]
            },
        }
        predictions1 = {
            'img': [],
            'data_sample': {
                'pred_alpha': masked_pred_alpha[0:1]
            },
        }

        # process 2 batches
        conn_err.process(data_batch, predictions)
        conn_err.process(data_batch1, predictions1)

        assert conn_err.results == [
            {
                'conn_err': 0.256,
                # 'n': 1
            },
            {
                'conn_err': 0.256,
                # 'n': 1
            },
            {
                'conn_err': 0.256,
                # 'n': 1
            },
        ]

        res = conn_err.compute_metrics(conn_err.results)

        assert list(res.keys()) == ['ConnectivityError']
        assert np.allclose(res['ConnectivityError'], 0.256)
