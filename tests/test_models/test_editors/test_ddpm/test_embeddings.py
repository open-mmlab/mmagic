# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmagic.models.editors.ddpm.embeddings import TimestepEmbedding, Timesteps


def test_TimestepEmbedding():
    input = torch.rand((1, 64, 16))
    timestep_emb = TimestepEmbedding(
        in_channels=16, time_embed_dim=16, act_fn='mish')
    output = timestep_emb.forward(input)
    assert output.shape == (1, 64, 16)

    timestep_emb = TimestepEmbedding(
        in_channels=16, time_embed_dim=16, out_dim=96)
    timestep_emb.act = None
    output = timestep_emb.forward(input)
    assert output.shape == (1, 64, 96)


def test_Timesteps():
    input = torch.tensor([4])
    timesteps = Timesteps(num_channels=9)
    emb = timesteps.forward(input)
    assert emb.shape == (1, 9)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
