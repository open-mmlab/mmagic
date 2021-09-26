import numpy as np
import torch
import torch.nn as nn

from mmedit.models import build_backbone


def test_dfd_net():

    model_cfg = dict(type='DFDNet', mid_channels=64)

    # build model
    model = build_backbone(model_cfg)

    # test attributes
    assert model.__class__.__name__ == 'DFDNet'

    # prepare data
    inputs = torch.rand(1, 3, 512, 512)
    part_locations = dict(
        left_eye=torch.tensor([[146, 184, 225, 263]]),
        right_eye=torch.tensor([[283, 179, 374, 270]]),
        nose=torch.tensor([[229, 296, 282, 349]]),
        mouth=torch.tensor([[229, 296, 282, 349]]))
    targets = torch.rand(1, 3, 512, 512)

    dictionary = dict()
    parts = ['left_eye', 'right_eye', 'nose', 'mouth']
    part_sizes = np.array([80, 80, 50, 110])
    channel_sizes = np.array([128, 256, 512, 512])

    for j, size in enumerate([256, 128, 64, 32]):
        dictionary[size] = dict()
        for i, part in enumerate(parts):
            dictionary[size][part] = torch.rand(512, channel_sizes[j],
                                                part_sizes[i] // (2**(j + 1)),
                                                part_sizes[i] // (2**(j + 1)))

    # prepare loss
    loss_function = nn.L1Loss()

    # prepare optimizer
    optimizer = torch.optim.Adam(model.parameters())

    # test on cpu
    output = model(inputs, part_locations, dictionary)
    optimizer.zero_grad()
    loss = loss_function(output, targets)
    loss.backward()
    optimizer.step()
    assert torch.is_tensor(output)
    assert output.shape == targets.shape

    # test on gpu
    if torch.cuda.is_available():
        model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters())
        inputs = inputs.cuda()
        targets = targets.cuda()
        output = model(inputs, part_locations, dictionary)
        optimizer.zero_grad()
        loss = loss_function(output, targets)
        loss.backward()
        optimizer.step()
        assert torch.is_tensor(output)
        assert output.shape == targets.shape
