import os.path as osp
import time
from tempfile import TemporaryDirectory

import mmcv
import torch
from mmcv.parallel import is_module_wrapper
from mmcv.runner.checkpoint import get_state_dict, weights_to_cpu
from torch.optim import Optimizer


def save_checkpoint(model,
                    filename,
                    optimizer=None,
                    loss_scaler=None,
                    save_apex_amp=False,
                    meta=None):
    """Save checkpoint to file.
    The checkpoint will have 3 or more fields: ``meta``, ``state_dict`` and
    ``optimizer``. By default ``meta`` will contain version and time info.
    In mixed-precision training, ``loss_scaler`` or ``amp.state_dict`` will be
    saved in checkpoint.
    Args:
        model (Module): Module whose params are to be saved.
        filename (str): Checkpoint filename.
        optimizer (:obj:`Optimizer`, optional): Optimizer to be saved.
        loss_scaler (Object, optional): Loss scaler used for FP16 training.
        save_apex_amp (bool, optional): Whether to save apex.amp
            ``state_dict``.
        meta (dict, optional): Metadata to be saved in checkpoint.
    """
    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        raise TypeError(f'meta must be a dict or None, but got {type(meta)}')
    meta.update(mmcv_version=mmcv.__version__, time=time.asctime())

    if is_module_wrapper(model):
        model = model.module

    if hasattr(model, 'CLASSES') and model.CLASSES is not None:
        # save class name to the meta
        meta.update(CLASSES=model.CLASSES)

    checkpoint = {
        'meta': meta,
        'state_dict': weights_to_cpu(get_state_dict(model))
    }
    # save optimizer state dict in the checkpoint
    if isinstance(optimizer, Optimizer):
        checkpoint['optimizer'] = optimizer.state_dict()
    elif isinstance(optimizer, dict):
        checkpoint['optimizer'] = {}
        for name, optim in optimizer.items():
            checkpoint['optimizer'][name] = optim.state_dict()

    # save loss scaler for mixed-precision (FP16) training
    if loss_scaler is not None:
        checkpoint['loss_scaler'] = loss_scaler.state_dict()

    # save state_dict from apex.amp
    if save_apex_amp:
        from apex import amp
        checkpoint['amp'] = amp.state_dict()

    if filename.startswith('pavi://'):
        try:
            from pavi import modelcloud
            from pavi.exception import NodeNotFoundError
        except ImportError:
            raise ImportError(
                'Please install pavi to load checkpoint from modelcloud.')
        model_path = filename[7:]
        root = modelcloud.Folder()
        model_dir, model_name = osp.split(model_path)
        try:
            model = modelcloud.get(model_dir)
        except NodeNotFoundError:
            model = root.create_training_model(model_dir)
        with TemporaryDirectory() as tmp_dir:
            checkpoint_file = osp.join(tmp_dir, model_name)
            with open(checkpoint_file, 'wb') as f:
                torch.save(checkpoint, f)
                f.flush()
            model.create_file(checkpoint_file, name=model_name)
    else:
        mmcv.mkdir_or_exist(osp.dirname(filename))
        # immediately flush buffer
        with open(filename, 'wb') as f:
            torch.save(checkpoint, f)
            f.flush()
