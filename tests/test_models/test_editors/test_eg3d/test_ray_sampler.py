# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmagic.models.editors.eg3d.ray_sampler import sample_rays


def test_sample_rays():
    cond1 = [
        0.5186675190925598, -0.8265361189842224, -0.21868225932121277,
        0.2842874228954315, 0.8549759387969971, 0.5014145970344543,
        0.13266240060329437, -0.17246174812316895, 2.3841852225814364e-07,
        -0.2557757496833801, 0.9667359590530396, -1.2567564249038696, -0.0,
        0.0, -0.0, 1.0, 1.025390625, 0.0, 0.5, 0.0, 1.025390625, 0.5, 0.0, 0.0,
        1.0
    ]
    cond2 = [
        0.828181266784668, -0.29340365529060364, -0.47752493619918823,
        0.6207822561264038, 0.5604602694511414, 0.43355682492256165,
        0.7056292295455933, -0.9173181653022766, -2.9802322387695312e-08,
        -0.8520227670669556, 0.5235047340393066, -0.680556058883667, -0.0, 0.0,
        -0.0, 1.0, 1.025390625, 0.0, 0.5, 0.0, 1.025390625, 0.5, 0.0, 0.0, 1.0
    ]
    cond1, cond2 = torch.FloatTensor(cond1), torch.FloatTensor(cond2)
    cond = torch.stack((cond1, cond2))
    cam2world = cond[:, :16].view(-1, 4, 4)
    intrinsics = cond[:, 16:25].view(-1, 3, 3)
    resolution = 16
    ray_origins, ray_directions = sample_rays(cam2world, intrinsics,
                                              resolution)
    assert ray_origins.shape[:2] == (2, resolution**2)
    assert ray_directions.shape[:2] == (2, resolution**2)
    # check if camera origin in one batch is all same
    for origin in ray_origins:
        assert (origin == origin[0]).all()


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
