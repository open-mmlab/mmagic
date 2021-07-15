import os.path as osp

from .base_generation_dataset import BaseGenerationDataset
from .registry import DATASETS


@DATASETS.register_module()
class GenerationPairedDataset(BaseGenerationDataset):
    """General paired image folder dataset for image generation.

    It assumes that the training directory is '/path/to/data/train'.
    During test time, the directory is '/path/to/data/test'. '/path/to/data'
    can be initialized by args 'dataroot'. Each sample contains a pair of
    images concatenated in the w dimension (A|B).

    Args:
        dataroot (str | :obj:`Path`): Path to the folder root of paired images.
        pipeline (List[dict | callable]): A sequence of data transformations.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
    """

    def __init__(self, dataroot, pipeline, test_mode=False):
        super().__init__(pipeline, test_mode)
        phase = 'test' if test_mode else 'train'
        self.dataroot = osp.join(str(dataroot), phase)
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        """Load paired image paths.

        Returns:
            list[dict]: List that contains paired image paths.
        """
        data_infos = []
        pair_paths = sorted(self.scan_folder(self.dataroot))
        for pair_path in pair_paths:
            data_infos.append(dict(pair_path=pair_path))

        return data_infos

@DATASETS.register_module()
class GenerationPairedBlurSharpDataset(BaseGenerationDataset):
    """General unpaired image folder dataset for image generation.

    It assumes that the training directory of images from domain A is
    '/path/to/data/trainA', and that from domain B is '/path/to/data/trainB',
    respectively. '/path/to/data' can be initialized by args 'dataroot'.
    During test time, the directory is '/path/to/data/testA' and
    '/path/to/data/testB', respectively.

    Args:
        dataroot (str | :obj:`Path`): Path to the folder root of unpaired
            images.
        pipeline (List[dict | callable]): A sequence of data transformations.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
    """

    def __init__(self, dataroot, pipeline, test_mode=False):
        super().__init__(pipeline, test_mode)
        phase = 'test' if test_mode else 'train'
        self.dataroot_a = osp.join(str(dataroot), phase + 'Blur')
        self.dataroot_b = osp.join(str(dataroot), phase + 'Sharp')
        self.data_infos_a = self.load_annotations(self.dataroot_a)
        self.data_infos_b = self.load_annotations(self.dataroot_b)
        self.len_a = len(self.data_infos_a)
        self.len_b = len(self.data_infos_b)
        assert self.len_a == self.len_b, print('must be paired')
    def load_annotations(self, dataroot):
        """Load unpaired image paths of one domain.

        Args:
            dataroot (str): Path to the folder root for unpaired images of
                one domain.

        Returns:
            list[dict]: List that contains unpaired image paths of one domain.
        """
        data_infos = []
        paths = sorted(self.scan_folder(dataroot))
        for path in paths:
            data_infos.append(dict(path=path))
        return data_infos

    def prepare_train_data(self, idx):
        """Prepare unpaired training data.

        Args:
            idx (int): Index of current batch.

        Returns:
            dict: Prepared training data batch.
        """
        img_a_path = self.data_infos_a[idx]['path']
        img_b_path = self.data_infos_b[idx]['path']
        results = dict(img_blur_real_path=img_a_path, img_sharp_real_path=img_b_path)
        return self.pipeline(results)

    def prepare_test_data(self, idx):
        """Prepare unpaired test data.

        Args:
            idx (int): Index of current batch.

        Returns:
            list[dict]: Prepared test data batch.
        """
        img_a_path = self.data_infos_a[idx]['path']
        img_b_path = self.data_infos_b[idx]['path']
        results = dict(img_blur_real_path=img_a_path, img_sharp_real_path=img_b_path)
        return self.pipeline(results)

    def __len__(self):
        return 1
        return max(self.len_a, self.len_b)