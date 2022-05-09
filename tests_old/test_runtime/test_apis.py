# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.apis.train import init_random_seed, set_random_seed


def test_init_random_seed():
    init_random_seed(0, device='cpu')
    init_random_seed(device='cpu')
    # test on gpu
    if torch.cuda.is_available():
        init_random_seed(0, device='cuda')
        init_random_seed(device='cuda')


def test_set_random_seed():
    set_random_seed(0, deterministic=False)
    set_random_seed(0, deterministic=True)
