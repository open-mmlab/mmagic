# Copyright (c) OpenMMLab. All rights reserved.
import copy
from collections import defaultdict

from .base_dataset import BaseDataset

IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF')


class BaseVFIDataset(BaseDataset):
    """Base class for video frame interpolation datasets."""

    def __init__(self, pipeline, folder, ann_file, test_mode=False):
        super().__init__(pipeline, test_mode)
        self.folder = str(folder)
        self.ann_file = str(ann_file)

    def __getitem__(self, idx):
        """Get item at each call.

        Args:
            idx (int): Index for getting each item.
        """
        results = copy.deepcopy(self.data_infos[idx])
        results['folder'] = self.folder
        results['ann_file'] = self.ann_file
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
        eval_result = {
            metric: sum(values) / len(self)
            for metric, values in eval_result.items()
        }

        return eval_result

    def load_annotations(self):
        """Abstract function for loading annotation.

        All subclasses should overwrite this function
        """
        pass
