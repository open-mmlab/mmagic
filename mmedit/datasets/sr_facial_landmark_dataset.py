# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import numpy as np

from .base_sr_dataset import BaseSRDataset
from .registry import DATASETS


@DATASETS.register_module()
class SRFacialLandmarkDataset(BaseSRDataset):
    """Facial image and landmark dataset with an annotation file for image
    restoration.

    The dataset loads gt (Ground-Truth) image, shape of image, face box, and
    landmark. Applies specified transforms and finally returns a dict
    containing paired data and other information.

    This is the "annotation file mode":
    Each dict in the annotation list contains the image names, image shape,
    face box, and landmark.

    Annotation file is a `npy` file, which contains a list of dict.
    Example of an annotation file:

    ::

        dict1(file=*, bbox=*, shape=*, landmark=*)
        dict2(file=*, bbox=*, shape=*, landmark=*)

    Args:
        gt_folder (str | :obj:`Path`): Path to a gt folder.
        ann_file (str | :obj:`Path`): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transformations.
        scale (int): Upsampling scale ratio.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
    """

    def __init__(self, gt_folder, ann_file, pipeline, scale, test_mode=False):
        super().__init__(pipeline, scale, test_mode)
        self.gt_folder = str(gt_folder)
        self.ann_file = str(ann_file)
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        """Load annotations for SR dataset.

        Annotation file is a `npy` file, which contains a list of dict.

        It loads the GT image path and landmark from the annotation file.
        Each dict in the annotation file contains the image names, image
        shape (usually for gt), bbox and landmark.

        Returns:
            list[dict]: A list of dicts for GT path and landmark.
                Contains: gt_path, bbox, shape, landmark.
        """
        data_infos = np.load(self.ann_file, allow_pickle=True)
        for data_info in data_infos:
            data_info['gt_path'] = osp.join(self.gt_folder,
                                            data_info['gt_path'])

        return data_infos
