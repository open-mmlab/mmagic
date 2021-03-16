import torch
import torch.nn as nn

from mmedit.models.builder import build_component


def test_mlp():
    model_cfg = dict(type='MLP', in_dim=8, out_dim=3, hidden_list=[8, 8, 8, 8])
    mlp = build_component(model_cfg)

    # test attributes
    assert mlp.__class__.__name__ == 'MLP'

    # prepare data
    inputs = torch.rand(2, 8)
    targets = torch.rand(2, 3)
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        targets = targets.cuda()
        mlp = mlp.cuda()
    data_batch = {'in': inputs, 'target': targets}
    # prepare optimizer
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

    # test train_step
    output = mlp.forward(data_batch['in'])
    assert output.shape == data_batch['target'].shape
    loss = criterion(output, data_batch['target'])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
