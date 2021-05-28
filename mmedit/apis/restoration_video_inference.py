import glob

import torch
from mmcv.parallel import collate, scatter

from mmedit.datasets.pipelines import Compose


def pad_sequence(data, window_size):
    padding = window_size // 2

    data = torch.cat([
        data[:, 1 + padding:1 + 2 * padding].flip(1), data,
        data[:, -1 - 2 * padding:-1 - padding].flip(1)
    ],
                     dim=1)

    return data


def restoration_video_inference(model, img_dir, window_size, start_idx,
                                filename_tmpl):
    """Inference image with the model.

    Args:
        model (nn.Module): The loaded model.
        img_dir (str): Directory of the input video.
        window_size (int): The window size used in sliding-window framework.
            This value should be set according to the settings of the network.
            A value smaller than 0 means using recurrent framework.
        start_idx (int): The index corresponds to the first frame in the
            sequence.
        filename_tmpl (str): Template for file name.

    Returns:
        Tensor: The predicted restoration result.
    """

    device = next(model.parameters()).device  # model device

    # pipeline
    test_pipeline = [
        dict(
            type='GenerateSegmentIndices',
            interval_list=[1],
            start_idx=start_idx,
            filename_tmpl=filename_tmpl),
        dict(
            type='LoadImageFromFileList',
            io_backend='disk',
            key='lq',
            channel_order='rgb'),
        dict(type='RescaleToZeroOne', keys=['lq']),
        dict(type='FramesToTensor', keys=['lq']),
        dict(type='Collect', keys=['lq'], meta_keys=['lq_path', 'key'])
    ]

    # build the data pipeline
    test_pipeline = Compose(test_pipeline)

    # prepare data
    sequence_length = len(glob.glob(f'{img_dir}/*'))
    key = img_dir.split('/')[-1]
    lq_folder = '/'.join(img_dir.split('/')[:-1])
    data = dict(
        lq_path=lq_folder,
        gt_path='',
        key=key,
        sequence_length=sequence_length)
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]['lq']

    # forward the model
    with torch.no_grad():
        if window_size > 0:  # sliding window framework
            data = pad_sequence(data, window_size)
            result = []
            for i in range(0, data.size(1) - 2 * window_size):
                data_i = data[:, i:i + window_size]
                result.append(model(lq=data_i, test_mode=True)['output'])
            result = torch.stack(result, dim=1)
        else:  # recurrent framework
            result = model(lq=data, test_mode=True)['output']

    return result
