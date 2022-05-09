# Copyright (c) OpenMMLab. All rights reserved.
from .base_sr_dataset import BaseSRDataset
from .registry import DATASETS


@DATASETS.register_module()
class SRFolderGTDataset(BaseSRDataset):
    """General ground-truth image folder dataset for image restoration.

    The dataset loads gt (Ground-Truth) image only,
    applies specified transforms and finally returns a dict containing paired
    data and other information.

    This is the "gt folder mode", which needs to specify the gt
    folder path, each folder containing the corresponding images.
    Image lists will be generated automatically.

    For example, we have a folder with the following structure:

    ::

        data_root
        ├── gt
        │   ├── 0001.png
        │   ├── 0002.png

    then, you need to set:

    .. code-block:: python

        gt_folder = data_root/gt

    Args:
        gt_folder (str | :obj:`Path`): Path to a gt folder.
        pipeline (List[dict | callable]): A sequence of data transformations.
        scale (int | tuple): Upsampling scale or upsampling scale range.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
    """

    def __init__(self,
                 gt_folder,
                 pipeline,
                 scale,
                 test_mode=False,
                 filename_tmpl='{}'):
        super().__init__(pipeline, scale, test_mode)
        self.gt_folder = str(gt_folder)
        self.filename_tmpl = filename_tmpl
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        """Load annotations for SR dataset.

        It loads the GT image path from folder.

        Returns:
            list[dict]: A list of dicts for path of GT.
        """
        data_infos = []
        gt_paths = self.scan_folder(self.gt_folder)
        for gt_path in gt_paths:
            data_infos.append(dict(gt_path=gt_path))
        return data_infos
