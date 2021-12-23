# Copyright (c) OpenMMLab. All rights reserved.
from torch.utils.data import Dataset

from mmedit.datasets import RepeatDataset


def test_repeat_dataset():

    class ToyDataset(Dataset):

        def __init__(self):
            super().__init__()
            self.members = [1, 2, 3, 4, 5]

        def __len__(self):
            return len(self.members)

        def __getitem__(self, idx):
            return self.members[idx % 5]

    toy_dataset = ToyDataset()
    repeat_dataset = RepeatDataset(toy_dataset, 2)
    assert len(repeat_dataset) == 10
    assert repeat_dataset[2] == 3
    assert repeat_dataset[8] == 4
