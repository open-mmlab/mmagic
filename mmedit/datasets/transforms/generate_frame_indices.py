# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp

import numpy as np
from mmcv.transforms import BaseTransform

from mmedit.registry import TRANSFORMS


@TRANSFORMS.register_module()
class GenerateFrameIndices(BaseTransform):
    """Generate frame index for REDS datasets. It also performs temporal
    augmention with random interval.

    Required Keys:

    - img_path
    - gt_path
    - key
    - num_input_frames

    Modified Keys:

    - img_path
    - gt_path

    Added Keys:

    - interval
    - reverse

    Args:
        interval_list (list[int]): Interval list for temporal augmentation.
            It will randomly pick an interval from interval_list and sample
            frame index with the interval.
        frames_per_clip(int): Number of frames per clips. Default: 99 for
            REDS dataset.
    """

    def __init__(self, interval_list, frames_per_clip=99):

        self.interval_list = interval_list
        self.frames_per_clip = frames_per_clip

    def transform(self, results):
        """transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        clip_name, frame_name = results['key'].split(
            os.sep)  # key example: 000/00000000
        center_frame_idx = int(frame_name)
        num_half_frames = results['num_input_frames'] // 2

        sequence_length = results.get('sequence_length',
                                      self.frames_per_clip + 1)
        frames_per_clip = min(self.frames_per_clip, sequence_length - 1)

        interval = np.random.choice(self.interval_list)
        # ensure not exceeding the borders
        start_frame_idx = center_frame_idx - num_half_frames * interval
        end_frame_idx = center_frame_idx + num_half_frames * interval
        while (start_frame_idx < 0) or (end_frame_idx > frames_per_clip):
            center_frame_idx = np.random.randint(0, frames_per_clip + 1)
            start_frame_idx = center_frame_idx - num_half_frames * interval
            end_frame_idx = center_frame_idx + num_half_frames * interval
        frame_name = f'{center_frame_idx:08d}'
        neighbor_list = list(
            range(center_frame_idx - num_half_frames * interval,
                  center_frame_idx + num_half_frames * interval + 1, interval))

        img_path_root = results['img_path']
        gt_path_root = results['gt_path']
        img_path = [
            osp.join(img_path_root, clip_name, f'{v:08d}.png')
            for v in neighbor_list
        ]
        gt_path = [osp.join(gt_path_root, clip_name, f'{frame_name}.png')]

        results['img_path'] = img_path
        results['gt_path'] = gt_path
        results['interval'] = interval

        return results

    def __repr__(self):

        repr_str = self.__class__.__name__
        repr_str += (f'(interval_list={self.interval_list}, '
                     f'frames_per_clip={self.frames_per_clip})')

        return repr_str


@TRANSFORMS.register_module()
class GenerateFrameIndiceswithPadding(BaseTransform):
    """Generate frame index with padding for REDS dataset and Vid4 dataset
    during testing.

    Required Keys:

    - img_path
    - gt_path
    - key
    - num_input_frames
    - sequence_length

    Modified Keys:

    - img_path
    - gt_path

    Args:
         padding (str): padding mode, one of
            'replicate' | 'reflection' | 'reflection_circle' | 'circle'.

            Examples: current_idx = 0, num_input_frames = 5
            The generated frame indices under different padding mode:

                replicate: [0, 0, 0, 1, 2]
                reflection: [2, 1, 0, 1, 2]
                reflection_circle: [4, 3, 0, 1, 2]
                circle: [3, 4, 0, 1, 2]

        filename_tmpl (str): Template for file name. Default: '{:08d}'.
    """

    def __init__(self, padding, filename_tmpl='{:08d}'):

        if padding not in ('replicate', 'reflection', 'reflection_circle',
                           'circle'):
            raise ValueError(f'Wrong padding mode {padding}.'
                             'Should be "replicate", "reflection", '
                             '"reflection_circle",  "circle"')
        self.padding = padding
        self.filename_tmpl = filename_tmpl

    def transform(self, results):
        """transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        clip_name, frame_name = results['key'].split(os.sep)
        current_idx = int(frame_name)
        sequence_length = results['sequence_length'] - 1  # start from 0
        num_input_frames = results['num_input_frames']
        num_pad = num_input_frames // 2

        frame_list = []
        for i in range(current_idx - num_pad, current_idx + num_pad + 1):
            if i < 0:
                if self.padding == 'replicate':
                    pad_idx = 0
                elif self.padding == 'reflection':
                    pad_idx = -i
                elif self.padding == 'reflection_circle':
                    pad_idx = current_idx + num_pad - i
                else:
                    pad_idx = num_input_frames + i
            elif i > sequence_length:
                if self.padding == 'replicate':
                    pad_idx = sequence_length
                elif self.padding == 'reflection':
                    pad_idx = sequence_length * 2 - i
                elif self.padding == 'reflection_circle':
                    pad_idx = (current_idx - num_pad) - (i - sequence_length)
                else:
                    pad_idx = i - num_input_frames
            else:
                pad_idx = i
            frame_list.append(pad_idx)

        img_path_root = results['img_path']
        gt_path_root = results['gt_path']
        img_paths = [
            osp.join(img_path_root, clip_name,
                     f'{self.filename_tmpl.format(idx)}.png')
            for idx in frame_list
        ]
        gt_paths = [osp.join(gt_path_root, clip_name, f'{frame_name}.png')]

        results['img_path'] = img_paths
        results['gt_path'] = gt_paths

        return results

    def __repr__(self):

        repr_str = self.__class__.__name__ + f"(padding='{self.padding}')"

        return repr_str


@TRANSFORMS.register_module()
class GenerateSegmentIndices(BaseTransform):
    """Generate frame indices for a segment. It also performs temporal
    augmention with random interval.

    Required Keys:

    - img_path
    - gt_path
    - key
    - num_input_frames
    - sequence_length

    Modified Keys:

    - img_path
    - gt_path

    Added Keys:

    - interval
    - reverse

    Args:
        interval_list (list[int]): Interval list for temporal augmentation.
            It will randomly pick an interval from interval_list and sample
            frame index with the interval.
        start_idx (int): The index corresponds to the first frame in the
            sequence. Default: 0.
        filename_tmpl (str): Template for file name. Default: '{:08d}.png'.
    """

    def __init__(self, interval_list, start_idx=0, filename_tmpl='{:08d}.png'):

        self.interval_list = interval_list
        self.filename_tmpl = filename_tmpl
        self.start_idx = start_idx

    def transform(self, results):
        """transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        # key example: '000', 'calendar' (sequence name)
        clip_name = results['key']
        interval = np.random.choice(self.interval_list)

        self.sequence_length = results['sequence_length']
        num_input_frames = results.get('num_input_frames',
                                       self.sequence_length)
        if num_input_frames is None:
            num_input_frames = self.sequence_length

        # randomly select a frame as start
        if self.sequence_length - num_input_frames * interval < 0:
            raise ValueError('The input sequence is not long enough to '
                             'support the current choice of [interval] or '
                             '[num_input_frames].')
        start_frame_idx = np.random.randint(
            0, self.sequence_length - num_input_frames * interval + 1)
        end_frame_idx = start_frame_idx + num_input_frames * interval
        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))
        neighbor_list = [v + self.start_idx for v in neighbor_list]

        # add the corresponding file paths
        img_path_root = results['img_path']
        gt_path_root = results['gt_path']
        img_path = [
            osp.join(img_path_root, clip_name, self.filename_tmpl.format(v))
            for v in neighbor_list
        ]
        gt_path = [
            osp.join(gt_path_root, clip_name, self.filename_tmpl.format(v))
            for v in neighbor_list
        ]

        results['img_path'] = img_path
        results['gt_path'] = gt_path
        results['interval'] = interval

        return results

    def __repr__(self):

        repr_str = self.__class__.__name__
        repr_str += (f'(interval_list={self.interval_list})')

        return repr_str
