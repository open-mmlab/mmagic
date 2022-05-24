# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch
import torch.nn as nn

from mmedit.models.common import SpatialTemporalEnsemble


def test_ensemble_cpu():
    model = nn.Identity()

    # spatial ensemble of an image
    ensemble = SpatialTemporalEnsemble(is_temporal_ensemble=False)
    inputs = torch.rand(1, 3, 4, 4)
    outputs = ensemble(inputs, model)
    np.testing.assert_almost_equal(inputs.numpy(), outputs.numpy())

    # spatial ensemble of a sequence
    ensemble = SpatialTemporalEnsemble(is_temporal_ensemble=False)
    inputs = torch.rand(1, 2, 3, 4, 4)
    outputs = ensemble(inputs, model)
    np.testing.assert_almost_equal(inputs.numpy(), outputs.numpy())

    # spatial and temporal ensemble of a sequence
    ensemble = SpatialTemporalEnsemble(is_temporal_ensemble=True)
    inputs = torch.rand(1, 2, 3, 4, 4)
    outputs = ensemble(inputs, model)
    np.testing.assert_almost_equal(inputs.numpy(), outputs.numpy())

    # spatial and temporal ensemble of an image
    with pytest.raises(ValueError):
        ensemble = SpatialTemporalEnsemble(is_temporal_ensemble=True)
        inputs = torch.rand(1, 3, 4, 4)
        outputs = ensemble(inputs, model)


def test_ensemble_cuda():
    if torch.cuda.is_available():
        model = nn.Identity().cuda()

        # spatial ensemble of an image
        ensemble = SpatialTemporalEnsemble(is_temporal_ensemble=False)
        inputs = torch.rand(1, 3, 4, 4).cuda()
        outputs = ensemble(inputs, model)
        np.testing.assert_almost_equal(inputs.cpu().numpy(),
                                       outputs.cpu().numpy())

        # spatial ensemble of a sequence
        ensemble = SpatialTemporalEnsemble(is_temporal_ensemble=False)
        inputs = torch.rand(1, 2, 3, 4, 4).cuda()
        outputs = ensemble(inputs, model)
        np.testing.assert_almost_equal(inputs.cpu().numpy(),
                                       outputs.cpu().numpy())

        # spatial and temporal ensemble of a sequence
        ensemble = SpatialTemporalEnsemble(is_temporal_ensemble=True)
        inputs = torch.rand(1, 2, 3, 4, 4).cuda()
        outputs = ensemble(inputs, model)
        np.testing.assert_almost_equal(inputs.cpu().numpy(),
                                       outputs.cpu().numpy())

        # spatial and temporal ensemble of an image
        with pytest.raises(ValueError):
            ensemble = SpatialTemporalEnsemble(is_temporal_ensemble=True)
            inputs = torch.rand(1, 3, 4, 4).cuda()
            outputs = ensemble(inputs, model)
