from pathlib import Path

from .base_dataset import BaseDataset
from .registry import DATASETS


@DATASETS.register_module
class ImgInpaintingDataset(BaseDataset):
    """Image dataset for inpainting.
    """

    def __init__(self, ann_file, pipeline, data_prefix=None, test_mode=False):
        super(ImgInpaintingDataset, self).__init__(pipeline, test_mode)
        self.ann_file = str(ann_file)
        self.data_prefix = str(data_prefix)
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        with open(self.ann_file, 'r') as f:
            img_infos = []
            for idx, line in enumerate(f):
                line = line.strip()
                _info = dict()
                line_split = line.split(' ')
                _info = dict(
                    gt_img_path=Path(self.data_prefix).joinpath(
                        line_split[0]).as_posix(),
                    gt_img_idx=idx)
                img_infos.append(_info)

        return img_infos
