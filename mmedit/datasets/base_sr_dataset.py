import copy
import os.path as osp
from pathlib import Path

from mmcv import scandir

from .base_dataset import BaseDataset

IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP')


class BaseSRDataset(BaseDataset):
    """Base class for super resolution datasets.
    """

    def __init__(self, pipeline, scale, test_mode=False):
        super(BaseSRDataset, self).__init__(pipeline, test_mode)
        self.scale = scale

    @staticmethod
    def scan_folder(path):
        """Obtain image path list (including sub-folders) from a given folder.

        Args:
            path (str | obj:`Path`): Folder path.

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
        results = copy.deepcopy(self.data_infos[idx])
        results['scale'] = self.scale
        return self.pipeline(results)
