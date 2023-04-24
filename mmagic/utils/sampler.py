# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import Optional

from mmengine.dataset import pseudo_collate
from mmengine.runner import Runner
from torch.utils.data import ConcatDataset, DataLoader


def _check_keys(sample_kwargs: dict, key: str) -> None:
    """Check whether target `key` is in the `sample_kwargs`."""
    assert key in sample_kwargs, (
        f'\'{key}\' must be set in \'sample_kwargs\'. But only find '
        f'following keys: \'{list(sample_kwargs.keys())}\'.')


def get_sampler(sample_kwargs: dict, runner: Optional[Runner]):
    """Get a sampler to loop input data.

    Args:
        sample_kwargs (dict): _description_
        runner (Optional[Runner]): _description_

    Returns:
        _type_: _description_
    """
    _check_keys(sample_kwargs, 'type')
    sampler_kwargs_ = deepcopy(sample_kwargs)
    sampler_type = sampler_kwargs_.pop('type')
    sampler = eval(f'{sampler_type}Sampler')(sampler_kwargs_, runner)
    return sampler


class ArgumentsSampler:
    """Dummy sampler only return input args multiple times."""

    def __init__(self,
                 sample_kwargs: dict,
                 runner: Optional[Runner] = None) -> None:
        _check_keys(sample_kwargs, 'max_times')
        assert isinstance(sample_kwargs['max_times'], int), (
            '\'max_times\' in \'sample_kwargs\' must be type of int.\'.')

        self.sample_kwargs = deepcopy(sample_kwargs)
        self.max_times = self.sample_kwargs.pop('max_times')
        self.forward_kwargs = self.sample_kwargs.pop('forward_kwargs')
        # set default num_batches from forward_kwargs
        self.forward_kwargs.setdefault('num_batches',
                                       self.sample_kwargs['num_batches'])
        self.idx = 0

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= self.max_times:
            raise StopIteration
        self.idx += 1
        return dict(inputs=deepcopy(self.forward_kwargs))


class NoiseSampler:
    """Noise sampler to by call `models.noise_fn` to generate noise."""

    def __init__(self, sample_kwargs: dict, runner: Runner) -> None:
        _check_keys(sample_kwargs, 'max_times')
        _check_keys(sample_kwargs, 'num_batches')

        self.sample_kwargs = deepcopy(sample_kwargs)
        self.max_times = self.sample_kwargs.pop('max_times')
        self.num_batches = self.sample_kwargs.pop('num_batches')

        module = runner.model
        if hasattr(module, 'module'):
            module = module.module
        self.module = module

        self.idx = 0

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= self.max_times:
            raise StopIteration
        self.idx += 1

        noise = self.module.noise_fn(num_batches=self.num_batches)
        sample_kwargs = deepcopy(self.sample_kwargs)
        sample_kwargs['noise'] = noise
        # return sample_kwargs
        return dict(inputs=sample_kwargs)


class DataSampler:
    """Sampler loop the train_dataloader."""

    def __init__(self, sample_kwargs: dict, runner: Runner) -> None:
        _check_keys(sample_kwargs, 'max_times')

        self.sample_kwargs = deepcopy(sample_kwargs)
        self.max_times = self.sample_kwargs.pop('max_times')

        # build a new vanilla dataloader, because we should not reset the one
        # used in the training process.
        dataset = runner.train_dataloader.dataset
        batch_size = runner.train_dataloader.batch_size
        self._dataloader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=pseudo_collate)
        self._iterator = iter(self._dataloader)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= self.max_times:
            self._iterator = iter(self._dataloader)
            raise StopIteration
        self.idx += 1
        return next(self._iterator)


class ValDataSampler:
    """Sampler loop the val_dataloader."""

    def __init__(self, sample_kwargs: dict, runner: Runner) -> None:
        _check_keys(sample_kwargs, 'max_times')

        self.sample_kwargs = deepcopy(sample_kwargs)
        self.max_times = self.sample_kwargs.pop('max_times')

        # build a new vanilla dataloader, because we should not reset the one
        # used in the training process.
        if hasattr(runner.val_loop, 'dataloader'):
            dataset = runner.val_loop.dataloader.dataset
            batch_size = runner.val_loop.dataloader.batch_size
        else:
            # MultiValLoop use `dataloaders` instead `dataloader`
            loaders = runner.val_loop.dataloaders
            dataset = ConcatDataset([loader.dataset for loader in loaders])
            batch_size = loaders[0].batch_size

        self._dataloader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=pseudo_collate)
        self._iterator = iter(self._dataloader)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= self.max_times:
            self._iterator = iter(self._dataloader)
            raise StopIteration
        self.idx += 1
        return next(self._iterator)
