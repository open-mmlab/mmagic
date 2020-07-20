import logging
import tempfile
from unittest.mock import MagicMock

import mmcv.runner
import pytest
import torch
import torch.nn as nn
from mmcv.runner import obj_from_dict
from torch.utils.data import DataLoader, Dataset

from mmedit.core import EvalIterHook


class ExampleDataset(Dataset):

    def __getitem__(self, idx):
        results = dict(imgs=torch.tensor([1]))
        return results

    def __len__(self):
        return 1


class ExampleModel(nn.Module):

    def __init__(self):
        super(ExampleModel, self).__init__()
        self.test_cfg = None
        self.conv = nn.Conv2d(3, 3, 3)

    def forward(self, imgs, test_mode=False, **kwargs):
        return imgs

    def train_step(self, data_batch, optimizer):
        rlt = self.forward(data_batch)
        return dict(result=rlt)


def test_eval_hook():
    with pytest.raises(TypeError):
        test_dataset = ExampleModel()
        data_loader = [
            DataLoader(
                test_dataset,
                batch_size=1,
                sampler=None,
                num_worker=0,
                shuffle=False)
        ]
        EvalIterHook(data_loader)

    test_dataset = ExampleDataset()
    test_dataset.evaluate = MagicMock(return_value=dict(test='success'))
    loader = DataLoader(test_dataset, batch_size=1)
    model = ExampleModel()
    data_loader = DataLoader(
        test_dataset, batch_size=1, sampler=None, num_workers=0, shuffle=False)
    eval_hook = EvalIterHook(data_loader)
    optim_cfg = dict(type='Adam', lr=2e-4, betas=(0.9, 0.999))
    optimizer = obj_from_dict(optim_cfg, torch.optim,
                              dict(params=model.parameters()))
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = mmcv.runner.IterBasedRunner(
            model=model,
            optimizer=optimizer,
            work_dir=tmpdir,
            logger=logging.getLogger())
        runner.register_hook(eval_hook)
        runner.run([loader], [('train', 1)], 1)
        test_dataset.evaluate.assert_called_with([torch.tensor([1])],
                                                 logger=runner.logger)
