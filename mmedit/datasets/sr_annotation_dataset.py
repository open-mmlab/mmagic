# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .base_sr_dataset import BaseSRDataset
from .registry import DATASETS


@DATASETS.register_module()
class SRAnnotationDataset(BaseSRDataset):
    """General paired image dataset with an annotation file for image
    restoration.

    The dataset loads lq (Low Quality) and gt (Ground-Truth) image pairs,
    applies specified transforms and finally returns a dict containing paired
    data and other information.

    This is the "annotation file mode":
    Each line in the annotation file contains the image names and
    image shape (usually for gt), separated by a white space.

    Example of an annotation file:

    ::

        0001_s001.png (480,480,3)
        0001_s002.png (480,480,3)

    Args:
        lq_folder (str | :obj:`Path`): Path to a lq folder.
        gt_folder (str | :obj:`Path`): Path to a gt folder.
        ann_file (str | :obj:`Path`): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transformations.
        scale (int): Upsampling scale ratio.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Default: '{}'.
    """

    def __init__(self,
                 lq_folder,
                 gt_folder,
                 ann_file,
                 pipeline,
                 scale,
                 test_mode=False,
                 filename_tmpl='{}'):
        super().__init__(pipeline, scale, test_mode)
        self.lq_folder = str(lq_folder)
        self.gt_folder = str(gt_folder)
        self.ann_file = str(ann_file)
        self.filename_tmpl = filename_tmpl
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        """Load annotations for SR dataset.

        It loads the LQ and GT image path from the annotation file.
        Each line in the annotation file contains the image names and
        image shape (usually for gt), separated by a white space.

        Returns:
            list[dict]: A list of dicts for paired paths of LQ and GT.
        """
        data_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                gt_name = line.split(' ')[0]
                basename, ext = osp.splitext(osp.basename(gt_name))
                lq_name = f'{self.filename_tmpl.format(basename)}{ext}'
                data_infos.append(
                    dict(
                        lq_path=osp.join(self.lq_folder, lq_name),
                        gt_path=osp.join(self.gt_folder, gt_name)))
        return data_infos
