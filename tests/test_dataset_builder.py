import math

from torch.utils.data import ConcatDataset, RandomSampler, SequentialSampler

from mmedit.datasets import (DATASETS, RepeatDataset, build_dataloader,
                             build_dataset)
from mmedit.datasets.samplers import DistributedSampler


@DATASETS.register_module()
class ToyDataset(object):

    def __init__(self, ann_file=None, cnt=0):
        self.ann_file = ann_file
        self.cnt = cnt

    def __item__(self, idx):
        return idx

    def __len__(self):
        return 100


@DATASETS.register_module()
class ToyDatasetWithAnnFile(object):

    def __init__(self, ann_file):
        self.ann_file = ann_file

    def __item__(self, idx):
        return idx

    def __len__(self):
        return 100


def test_build_dataset():
    cfg = dict(type='ToyDataset')

    dataset = build_dataset(cfg)
    assert isinstance(dataset, ToyDataset)
    assert dataset.cnt == 0

    # test default_args
    dataset = build_dataset(cfg, default_args=dict(cnt=1))
    assert isinstance(dataset, ToyDataset)
    assert dataset.cnt == 1

    # test RepeatDataset
    cfg = dict(type='RepeatDataset', dataset=dict(type='ToyDataset'), times=3)
    dataset = build_dataset(cfg)
    assert isinstance(dataset, RepeatDataset)
    assert isinstance(dataset.dataset, ToyDataset)
    assert dataset.times == 3

    # test when ann_file is a list
    cfg = dict(
        type='ToyDatasetWithAnnFile', ann_file=['ann_file_a', 'ann_file_b'])
    dataset = build_dataset(cfg)
    assert isinstance(dataset, ConcatDataset)
    assert isinstance(dataset.datasets, list)
    assert isinstance(dataset.datasets[0], ToyDatasetWithAnnFile)
    assert dataset.datasets[0].ann_file == 'ann_file_a'
    assert isinstance(dataset.datasets[1], ToyDatasetWithAnnFile)
    assert dataset.datasets[1].ann_file == 'ann_file_b'

    # test concat dataset
    cfg = (dict(type='ToyDataset'),
           dict(type='ToyDatasetWithAnnFile', ann_file='ann_file'))
    dataset = build_dataset(cfg)
    assert isinstance(dataset, ConcatDataset)
    assert isinstance(dataset.datasets, list)
    assert isinstance(dataset.datasets[0], ToyDataset)
    assert isinstance(dataset.datasets[1], ToyDatasetWithAnnFile)


def test_build_dataloader():
    dataset = ToyDataset()
    samples_per_gpu = 3
    # dist=True, shuffle=True, 1GPU
    dataloader = build_dataloader(
        dataset, samples_per_gpu=samples_per_gpu, workers_per_gpu=2)
    assert dataloader.batch_size == samples_per_gpu
    assert len(dataloader) == int(math.ceil(len(dataset) / samples_per_gpu))
    assert isinstance(dataloader.sampler, DistributedSampler)
    assert dataloader.sampler.shuffle

    # dist=True, shuffle=False, 1GPU
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=2,
        shuffle=False)
    assert dataloader.batch_size == samples_per_gpu
    assert len(dataloader) == int(math.ceil(len(dataset) / samples_per_gpu))
    assert isinstance(dataloader.sampler, DistributedSampler)
    assert not dataloader.sampler.shuffle

    # dist=True, shuffle=True, 8GPU
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=2,
        num_gpus=8)
    assert dataloader.batch_size == samples_per_gpu
    assert len(dataloader) == int(math.ceil(len(dataset) / samples_per_gpu))
    assert dataloader.num_workers == 2

    # dist=False, shuffle=True, 1GPU
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=2,
        dist=False)
    assert dataloader.batch_size == samples_per_gpu
    assert len(dataloader) == int(math.ceil(len(dataset) / samples_per_gpu))
    assert isinstance(dataloader.sampler, RandomSampler)
    assert dataloader.num_workers == 2

    # dist=False, shuffle=False, 1GPU
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=3,
        workers_per_gpu=2,
        shuffle=False,
        dist=False)
    assert dataloader.batch_size == samples_per_gpu
    assert len(dataloader) == int(math.ceil(len(dataset) / samples_per_gpu))
    assert isinstance(dataloader.sampler, SequentialSampler)
    assert dataloader.num_workers == 2

    # dist=False, shuffle=True, 8GPU
    dataloader = build_dataloader(
        dataset, samples_per_gpu=3, workers_per_gpu=2, num_gpus=8, dist=False)
    assert dataloader.batch_size == samples_per_gpu * 8
    assert len(dataloader) == int(
        math.ceil(len(dataset) / samples_per_gpu / 8))
    assert isinstance(dataloader.sampler, RandomSampler)
    assert dataloader.num_workers == 16
