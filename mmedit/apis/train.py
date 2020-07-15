import os
import os.path as osp
import random

import mmcv
import numpy as np
import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import HOOKS, IterBasedRunner
from mmedit.core import DistEvalIterHook, EvalIterHook, build_optimizers
from mmedit.core.distributed_wrapper import DistributedDataParallelWrapper
from mmedit.datasets.builder import build_dataloader, build_dataset
from mmedit.utils import get_root_logger


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
    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            dist=True,
            drop_last=cfg.data.get('drop_last', False),
            seed=cfg.seed) for ds in dataset
    ]
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
        samples_per_gpu = cfg.data.get('val_samples_per_gpu',
                                       cfg.data.samples_per_gpu)
        workers_per_gpu = cfg.data.get('val_workers_per_gpu',
                                       cfg.data.workers_per_gpu)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=workers_per_gpu,
            dist=True,
            shuffle=False)
        save_path = osp.join(cfg.work_dir, 'val_visuals')
        runner.register_hook(
            DistEvalIterHook(
                data_loader, save_path=save_path, **cfg.evaluation))

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
    if validate:
        raise NotImplementedError('Built-in validation is not implemented '
                                  'yet in not-distributed training. Use '
                                  'distributed training or test.py and '
                                  '*eval.py scripts instead.')
    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            cfg.gpus,
            dist=False,
            drop_last=cfg.data.get('drop_last', False),
            seed=cfg.seed) for ds in dataset
    ]
    # put model on gpus
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()

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
        samples_per_gpu = cfg.data.get('val_samples_per_gpu',
                                       cfg.data.samples_per_gpu)
        workers_per_gpu = cfg.data.get('val_workers_per_gpu',
                                       cfg.data.workers_per_gpu)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=workers_per_gpu,
            dist=True,
            shuffle=False)
        save_path = osp.join(cfg.work_dir, 'val_visuals')
        runner.register_hook(
            EvalIterHook(data_loader, save_path=save_path, **cfg.evaluation))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_iters)
