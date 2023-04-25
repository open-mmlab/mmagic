# Copyright (c) OpenMMLab. All rights reserved.
import math

import mmcv
import numpy as np
import torch
from mmengine import Config
from mmengine.config import ConfigDict
from mmengine.fileio import get_file_backend
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmengine.runner import set_random_seed as set_random_seed_engine

from mmagic.registry import MODELS

VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi')
FILE_CLIENT = get_file_backend(backend_args={'backend': 'local'})


def set_random_seed(seed, deterministic=False, use_rank_shift=True):
    """Set random seed.

    In this function, we just modify the default behavior of the similar
    function defined in MMCV.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
        rank_shift (bool): Whether to add rank number to the random seed to
            have different random seed in different threads. Default: True.
    """
    set_random_seed_engine(seed, deterministic, use_rank_shift)


def delete_cfg(cfg, key='init_cfg'):
    """Delete key from config object.

    Args:
        cfg (str or :obj:`mmengine.Config`): Config object.
        key (str): Which key to delete.
    """

    if key in cfg:
        cfg.pop(key)
    for _key in cfg.keys():
        if isinstance(cfg[_key], ConfigDict):
            delete_cfg(cfg[_key], key)


def init_model(config, checkpoint=None, device='cuda:0'):
    """Initialize a model from config file.

    Args:
        config (str or :obj:`mmengine.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str): Which device the model will deploy. Default: 'cuda:0'.

    Returns:
        nn.Module: The constructed model.
    """

    if isinstance(config, str):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    # config.test_cfg.metrics = None
    delete_cfg(config.model, 'init_cfg')

    init_default_scope(config.get('default_scope', 'mmagic'))

    model = MODELS.build(config.model)

    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)

    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()

    return model


def pad_sequence(data, window_size):
    """Pad frame sequence data.

    Args:
        data (Tensor): The frame sequence data.
        window_size (int): The window size used in sliding-window framework.

    Returns:
        data (Tensor): The padded result.
    """

    padding = window_size // 2

    data = torch.cat([
        data[:, 1 + padding:1 + 2 * padding].flip(1), data,
        data[:, -1 - 2 * padding:-1 - padding].flip(1)
    ],
                     dim=1)

    return data


def read_image(filepath):
    """Read image from file.

    Args:
        filepath (str): File path.

    Returns:
        image (np.array): Image.
    """
    img_bytes = FILE_CLIENT.get(filepath)
    image = mmcv.imfrombytes(
        img_bytes, flag='color', channel_order='rgb', backend='pillow')
    return image


def read_frames(source, start_index, num_frames, from_video, end_index):
    """Read frames from file or video.

    Args:
        source (list | mmcv.VideoReader): Source of frames.
        start_index (int): Start index of frames.
        num_frames (int): frames number to be read.
        from_video (bool): Weather read frames from video.
        end_index (int): The end index of frames.

    Returns:
        images (np.array): Images.
    """
    images = []
    last_index = min(start_index + num_frames, end_index)
    # read frames from video
    if from_video:
        for index in range(start_index, last_index):
            if index >= source.frame_cnt:
                break
            images.append(np.flip(source.get_frame(index), axis=2))
    else:
        files = source[start_index:last_index]
        images = [read_image(f) for f in files]
    return images


def calculate_grid_size(num_batches: int = 1, aspect_ratio: int = 1) -> int:
    """Calculate the number of images per row (nrow) to make the grid closer to
    square when formatting a batch of images to grid.

    Args:
        num_batches (int, optional): Number of images per batch. Defaults to 1.
        aspect_ratio (int, optional): The aspect ratio (width / height) of
            each image sample. Defaults to 1.

    Returns:
        int: Calculated number of images per row.
    """
    curr_ncol, curr_nrow = 1, num_batches
    curr_delta = curr_nrow * aspect_ratio - curr_ncol

    nrow = curr_nrow
    delta = curr_delta

    while curr_delta > 0:

        curr_ncol += 1
        curr_nrow = math.ceil(num_batches / curr_ncol)

        curr_delta = curr_nrow * aspect_ratio - curr_ncol
        if curr_delta < delta and curr_delta >= 0:
            nrow, delta = curr_nrow, curr_delta

    return nrow
