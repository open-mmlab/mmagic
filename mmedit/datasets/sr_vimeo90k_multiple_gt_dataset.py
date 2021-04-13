import glob
import os.path as osp

from .base_sr_dataset import BaseSRDataset
from .registry import DATASETS


@DATASETS.register_module()
class SRVimeo90KMultipleGTDataset(BaseSRDataset):
    """Vimeo90K dataset for video super resolution for recurrent networks.

    The dataset loads several LQ (Low-Quality) frames and GT (Ground-Truth)
    frames. Then it applies specified transforms and finally returns a dict
    containing paired data and other information.

    It reads Vimeo90K keys from the txt file. Each line contains:

        1. video frame folder
        2. image shape

    Examples:

    ::

        00001/0266 (256,448,3)
        00001/0268 (256,448,3)

    Args:
        lq_folder (str | :obj:`Path`): Path to a lq folder.
        gt_folder (str | :obj:`Path`): Path to a gt folder.
        ann_file (str | :obj:`Path`): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transformations.
        scale (int): Upsampling scale ratio.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
    """

    def __init__(self, lq_folder, gt_folder, ann_file, pipeline, scale,
                 test_mode):
        super().__init__(pipeline, scale, test_mode)

        self.lq_folder = str(lq_folder)
        self.gt_folder = str(gt_folder)
        self.ann_file = str(ann_file)

        self.data_infos = self.load_annotations()

    def load_annotations(self):
        """Load annoations for Vimeo-90K dataset.

        Returns:
            dict: Returned dict for LQ and GT pairs.
        """
        # get keys
        with open(self.ann_file, 'r') as fin:
            keys = [line.strip().split(' ')[0] for line in fin]

        data_infos = []
        for key in keys:
            lq_paths = sorted(
                glob.glob(osp.join(self.lq_folder, key, '*.png')))
            gt_paths = sorted(
                glob.glob(osp.join(self.gt_folder, key, '*.png')))

            data_infos.append(
                dict(lq_path=lq_paths, gt_path=gt_paths, key=key))

        return data_infos
