# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.models.editors.ddpm.embeddings import (TimestepEmbedding,
                                                   get_timestep_embedding)


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


def test_get_timestep_embedding():
    input = torch.tensor([4])
    emb = get_timestep_embedding(input, embedding_dim=9)
    assert emb.shape == (1, 9)


if __name__ == '__main__':
    test_TimestepEmbedding()
