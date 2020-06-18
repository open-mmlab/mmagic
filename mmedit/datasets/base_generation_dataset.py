import os.path as osp
from pathlib import Path

from mmcv import scandir

from .base_dataset import BaseDataset

IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF')


class BaseGenerationDataset(BaseDataset):
    """Base class for generation datasets."""

    @staticmethod
    def scan_folder(path):
        """Obtain image path list (including sub-folders) from a given folder.

        Args:
            path (str | :obj:`Path`): Folder path.

        Returns:
            list[str]: Image list obtained from the given folder.
        """

        if isinstance(path, (str, Path)):
            path = str(path)
        else:
            raise TypeError("'path' must be a str or a Path object, "
                            f'but received {type(path)}.')

        images = scandir(path, suffix=IMG_EXTENSIONS, recursive=True)
        images = [osp.join(path, v) for v in images]
        assert images, f'{path} has no valid image file.'
        return images

    def evaluate(self, results, logger=None):
        """Evaluating with saving generated images. (needs no metrics)

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

        results = [res['saved_flag'] for res in results]
        saved_num = 0
        for flag in results:
            if flag:
                saved_num += 1

        # make a dict to show
        eval_results = {'val_saved_number': saved_num}

        return eval_results
