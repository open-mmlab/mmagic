# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from collections import defaultdict
from pathlib import Path

import numpy as np
from mmcv import scandir

from mmedit.core.registry import build_metric
from .base_dataset import BaseDataset

IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF')
FEATURE_BASED_METRICS = ['FID', 'KID']


class BaseSRDataset(BaseDataset):
    """Base class for super resolution datasets."""

    def __init__(self, pipeline, scale, test_mode=False):
        super().__init__(pipeline, test_mode)
        self.scale = scale

    @staticmethod
    def scan_folder(path):
        """Obtain image path list (including sub-folders) from a given folder.

        Args:
            path (str | :obj:`Path`): Folder path.

        Returns:
            list[str]: image list obtained form given folder.
        """

        if isinstance(path, (str, Path)):
            path = str(path)
        else:
            raise TypeError("'path' must be a str or a Path object, "
                            f'but received {type(path)}.')

        images = list(scandir(path, suffix=IMG_EXTENSIONS, recursive=True))
        images = [osp.join(path, v) for v in images]
        assert images, f'{path} has no valid image file.'
        return images

    def __getitem__(self, idx):
        """Get item at each call.

        Args:
            idx (int): Index for getting each item.
        """
        results = copy.deepcopy(self.data_infos[idx])
        results['scale'] = self.scale
        return self.pipeline(results)

    def evaluate(self, results, logger=None):
        """Evaluate with different metrics.

        Args:
            results (list[tuple]): The output of forward_test() of the model.

        Return:
            dict: Evaluation results dict.
        """
        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        results = [res['eval_result'] for res in results]  # a list of dict
        eval_result = defaultdict(list)  # a dict of list

        for res in results:
            for metric, val in res.items():
                eval_result[metric].append(val)
        for metric, val_list in eval_result.items():
            assert len(val_list) == len(self), (
                f'Length of evaluation result of {metric} is {len(val_list)}, '
                f'should be {len(self)}')

        # average the results
        eval_result.update({
            metric: sum(values) / len(self)
            for metric, values in eval_result.items()
            if metric not in ['_inception_feat'] + FEATURE_BASED_METRICS
        })

        # evaluate feature-based metrics
        if '_inception_feat' in eval_result:
            feat1, feat2 = [], []
            for f1, f2 in eval_result['_inception_feat']:
                feat1.append(f1)
                feat2.append(f2)
            feat1 = np.concatenate(feat1, 0)
            feat2 = np.concatenate(feat2, 0)

            for metric in FEATURE_BASED_METRICS:
                if metric in eval_result:
                    metric_func = build_metric(eval_result[metric].pop())
                    eval_result[metric] = metric_func(feat1, feat2)

            # delete a redundant key for clean logging
            del eval_result['_inception_feat']

        return eval_result
