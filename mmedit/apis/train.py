# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import random
import warnings

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel
from mmcv.runner import HOOKS, IterBasedRunner, get_dist_info
from mmcv.utils import build_from_cfg

from mmedit.core import DistEvalIterHook, EvalIterHook, build_optimizers
from mmedit.core.distributed_wrapper import DistributedDataParallelWrapper
from mmedit.datasets.builder import build_dataloader, build_dataset
from mmedit.utils import get_root_logger


def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                meta=None):
    """Train model entry function.

    Args:
        model (nn.Module): The model to be trained.
        dataset (:obj:`Dataset`): Train dataset.
        cfg (dict): The config dict for training.
        distributed (bool): Whether to use distributed training.
            Default: False.
        validate (bool): Whether to do evaluation. Default: False.
        timestamp (str | None): Local time for runner. Default: None.
        meta (dict | None): Meta dict to record some important information.
            Default: None
    """
    logger = get_root_logger(log_level=cfg.log_level)

    # start training
    if distributed:
        _dist_train(
            model,
            dataset,
            cfg,
            validate=validate,
            logger=logger,
            timestamp=timestamp,
            meta=meta)
    else:
        _non_dist_train(
            model,
            dataset,
            cfg,
            validate=validate,
            logger=logger,
            timestamp=timestamp,
            meta=meta)


def _dist_train(model,
                dataset,
                cfg,
                validate=False,
                logger=None,
                timestamp=None,
                meta=None):
    """Distributed training function.

    Args:
        model (nn.Module): The model to be trained.
        dataset (:obj:`Dataset`): Train dataset.
        cfg (dict): The config dict for training.
        validate (bool): Whether to do evaluation. Default: False.
        logger (logging.Logger | None): Logger for training. Default: None.
        timestamp (str | None): Local time for runner. Default: None.
        meta (dict | None): Meta dict to record some important information.
            Default: None.
    """
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    # step 1: give default values and override (if exist) from cfg.data
    loader_cfg = {
        **dict(seed=cfg.get('seed'), drop_last=False, dist=True),
        **({} if torch.__version__ != 'parrots' else dict(
               prefetch_num=2,
               pin_memory=False,
           )),
        **dict((k, cfg.data[k]) for k in [
                   'samples_per_gpu',
                   'workers_per_gpu',
                   'shuffle',
                   'seed',
                   'drop_last',
                   'prefetch_num',
                   'pin_memory',
               ] if k in cfg.data)
    }

    # step 2: cfg.data.train_dataloader has highest priority
    train_loader_cfg = dict(loader_cfg, **cfg.data.get('train_dataloader', {}))

    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]

    # put model on gpus
    find_unused_parameters = cfg.get('find_unused_parameters', False)
    model = DistributedDataParallelWrapper(
        model,
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False,
        find_unused_parameters=find_unused_parameters)

    # build runner
    optimizer = build_optimizers(model, cfg.optimizers)
    runner = IterBasedRunner(
        model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta)
    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        checkpoint_config=cfg.checkpoint_config,
        log_config=cfg.log_config)

    # visual hook
    if cfg.get('visual_config', None) is not None:
        cfg.visual_config['output_dir'] = os.path.join(
            cfg.work_dir, cfg.visual_config['output_dir'])
        runner.register_hook(mmcv.build_from_cfg(cfg.visual_config, HOOKS))

    # evaluation hook
    if validate and cfg.get('evaluation', None) is not None:
        dataset = build_dataset(cfg.data.val)

        if ('val_samples_per_gpu' in cfg.data
                or 'val_workers_per_gpu' in cfg.data):
            warnings.warn('"val_samples_per_gpu/val_workers_per_gpu" have '
                          'been deprecated. Please use '
                          '"val_dataloader=dict(samples_per_gpu=1)" instead. '
                          'Details see '
                          'https://github.com/open-mmlab/mmediting/pull/201')

        val_loader_cfg = {
            **loader_cfg,
            **dict(shuffle=False, drop_last=False),
            **dict((newk, cfg.data[oldk]) for oldk, newk in [
                       ('val_samples_per_gpu', 'samples_per_gpu'),
                       ('val_workers_per_gpu', 'workers_per_gpu'),
                   ] if oldk in cfg.data),
            **cfg.data.get('val_dataloader', {})
        }

        data_loader = build_dataloader(dataset, **val_loader_cfg)
        save_path = osp.join(cfg.work_dir, 'val_visuals')
        runner.register_hook(
            DistEvalIterHook(
                data_loader, save_path=save_path, **cfg.evaluation),
            priority='LOW')

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_iters)


def _non_dist_train(model,
                    dataset,
                    cfg,
                    validate=False,
                    logger=None,
                    timestamp=None,
                    meta=None):
    """Non-Distributed training function.

    Args:
        model (nn.Module): The model to be trained.
        dataset (:obj:`Dataset`): Train dataset.
        cfg (dict): The config dict for training.
        validate (bool): Whether to do evaluation. Default: False.
        logger (logging.Logger | None): Logger for training. Default: None.
        timestamp (str | None): Local time for runner. Default: None.
        meta (dict | None): Meta dict to record some important information.
            Default: None.
    """
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    # step 1: give default values and override (if exist) from cfg.data
    loader_cfg = {
        **dict(
            seed=cfg.get('seed'),
            drop_last=False,
            dist=False,
            num_gpus=cfg.gpus),
        **({} if torch.__version__ != 'parrots' else dict(
               prefetch_num=2,
               pin_memory=False,
           )),
        **dict((k, cfg.data[k]) for k in [
                   'samples_per_gpu',
                   'workers_per_gpu',
                   'shuffle',
                   'seed',
                   'drop_last',
                   'prefetch_num',
                   'pin_memory',
               ] if k in cfg.data)
    }

    # step 2: cfg.data.train_dataloader has highest priority
    train_loader_cfg = dict(loader_cfg, **cfg.data.get('train_dataloader', {}))

    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]

    # put model on gpus/cpus
    model = MMDataParallel(model, device_ids=range(cfg.gpus))

    # build runner
    optimizer = build_optimizers(model, cfg.optimizers)
    runner = IterBasedRunner(
        model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta)

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        checkpoint_config=cfg.checkpoint_config,
        log_config=cfg.log_config)

    # visual hook
    if cfg.get('visual_config', None) is not None:
        cfg.visual_config['output_dir'] = os.path.join(
            cfg.work_dir, cfg.visual_config['output_dir'])
        runner.register_hook(mmcv.build_from_cfg(cfg.visual_config, HOOKS))

    # evaluation hook
    if validate and cfg.get('evaluation', None) is not None:
        dataset = build_dataset(cfg.data.val)

        if ('val_samples_per_gpu' in cfg.data
                or 'val_workers_per_gpu' in cfg.data):
            warnings.warn('"val_samples_per_gpu/val_workers_per_gpu" have '
                          'been deprecated. Please use '
                          '"val_dataloader=dict(samples_per_gpu=1)" instead. '
                          'Details see '
                          'https://github.com/open-mmlab/mmediting/pull/201')

        val_loader_cfg = {
            **loader_cfg,
            **dict(shuffle=False, drop_last=False),
            **dict((newk, cfg.data[oldk]) for oldk, newk in [
                       ('val_samples_per_gpu', 'samples_per_gpu'),
                       ('val_workers_per_gpu', 'workers_per_gpu'),
                   ] if oldk in cfg.data),
            **cfg.data.get('val_dataloader', {})
        }

        data_loader = build_dataloader(dataset, **val_loader_cfg)
        save_path = osp.join(cfg.work_dir, 'val_visuals')
        runner.register_hook(
            EvalIterHook(data_loader, save_path=save_path, **cfg.evaluation),
            priority='LOW')

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_iters)
