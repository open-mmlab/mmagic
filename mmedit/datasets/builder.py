import copy
import platform
import random
from functools import partial

import numpy as np
import torch
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import build_from_cfg
from torch.utils.data import ConcatDataset

from .dataset_wrappers import RepeatDataset
from .registry import DATASETS
from .samplers import DistributedSampler

if torch.__version__ == 'parrots':
    from torch.utils.data import PoolDataLoader
else:
    from torch.utils.data import DataLoader

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    hard_limit = rlimit[1]
    soft_limit = min(4096, hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))


def _concat_dataset(cfg, default_args=None):
    """Concat datasets with different ann_file but the same type.

    Args:
        cfg (dict): The config of dataset.
        default_args (dict, optional): Default initialization arguments.
            Default: None.

    Returns:
        Dataset: The concatenated dataset.
    """
    ann_files = cfg['ann_file']

    datasets = []
    num_dset = len(ann_files)
    for i in range(num_dset):
        data_cfg = copy.deepcopy(cfg)
        data_cfg['ann_file'] = ann_files[i]
        datasets.append(build_dataset(data_cfg, default_args))

    return ConcatDataset(datasets)


def build_dataset(cfg, default_args=None):
    """Build a dataset from config dict.

    It supports a variety of dataset config. If ``cfg`` is a Sequential (list
    or dict), it will be a concatenated dataset of the datasets specified by
    the Sequential. If it is a ``RepeatDataset``, then it will repeat the
    dataset ``cfg['dataset']`` for ``cfg['times']`` times. If the ``ann_file``
    of the dataset is a Sequential, then it will build a concatenated dataset
    with the same dataset type but different ``ann_file``.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict, optional): Default initialization arguments.
            Default: None.

    Returns:
        Dataset: The constructed dataset.
    """
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif isinstance(cfg.get('ann_file'), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     drop_last=False,
                     pin_memory=True,
                     **kwargs):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (:obj:`Dataset`): A PyTorch dataset.
        samples_per_gpu (int): Number of samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data
            loading for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed
            training. Default: 1.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        seed (int | None): Seed to be used. Default: None.
        drop_last (bool): Whether to drop the last incomplete batch in epoch.
            Default: False
        pin_memory (bool): Whether to use pin_memory in DataLoader.
            Default: True
        kwargs (dict, optional): Any keyword argument to be used to initialize
            DataLoader.

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    if torch.__version__ == 'parrots':
        prefetch_num = kwargs.get('prefetch_num', 2)
        DataLoader = partial(PoolDataLoader, prefetch_num=prefetch_num)

    rank, world_size = get_dist_info()
    if dist:
        sampler = DistributedSampler(
            dataset,
            world_size,
            rank,
            shuffle=shuffle,
            samples_per_gpu=samples_per_gpu)
        shuffle = False
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=pin_memory,
        shuffle=shuffle,
        worker_init_fn=init_fn,
        drop_last=drop_last,
        **kwargs)

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Function to initialize each worker.

    The seed of each worker equals to
    ``num_worker * rank + worker_id + user_seed``.

    Args:
        worker_id (int): Id for each worker.
        num_workers (int): Number of workers.
        rank (int): Rank in distributed training.
        seed (int): Random seed.
    """

    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
