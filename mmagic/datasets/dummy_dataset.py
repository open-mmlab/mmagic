# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

from mmedit.registry import DATASETS
from torch.utils.data import Dataset


@DATASETS.register_module()
class DummyDataset(Dataset):

    def __init__(self, max_length=100, batch_size=None, sample_kwargs=None):
        super().__init__()
        self.max_length = max_length
        self.sample_kwargs = sample_kwargs
        self.batch_size = batch_size

    def __len__(self):
        return self.max_length

    def __getitem__(self, index):
        data_dict = dict()
        input_dict = dict()
        if self.batch_size is not None:
            input_dict['num_batches'] = self.batch_size
        if self.sample_kwargs is not None:
            input_dict['sample_kwargs'] = deepcopy(self.sample_kwargs)

        data_dict['inputs'] = input_dict
        return data_dict
