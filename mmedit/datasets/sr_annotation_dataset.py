import os.path as osp

from .base_sr_dataset import BaseSRDataset
from .registry import DATASETS


@DATASETS.register_module
class SRAnnotationDataset(BaseSRDataset):
    """General paired image dataset with an annotation file for image
    restoration.

    The dataset loads lq (Low Quality) and gt (Ground-Truth) image pairs,
    applies specified transforms and finally returns a dict containing paired
    data and other information.

    This is the "annotation file mode":
    Annotation file is a txt file listing all paths of pairs.
    Each line contains the relative lq and gt paths, separated by a
    white space.

    Example of an annotation file:
    ```
    lq/0001_x4.png gt/0001.png
    lq/0002_x4.png gt/0002.png
    ```

    Args:
        ann_file (str | obj:`Path`): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transformations.
        scale (int): Upsampling scale ratio.
        data_prefix (str | obj:`Path`): Data root. Default: None.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 scale,
                 data_prefix=None,
                 test_mode=False):
        super(SRAnnotationDataset, self).__init__(pipeline, scale, test_mode)
        self.ann_file = str(ann_file)
        self.data_prefix = str(data_prefix)
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        data_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                lq_path, gt_path = line.strip().split(' ')
                data_infos.append(
                    dict(
                        lq_path=osp.join(self.data_prefix, lq_path),
                        gt_path=osp.join(self.data_prefix, gt_path)))
        return data_infos
