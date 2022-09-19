# Copyright (c) OpenMMLab. All rights reserved.

from pathlib import Path

from .base_dataset import BaseDataset
from .registry import DATASETS


@DATASETS.register_module()
class COCOStuff_Full_Dataset(BaseDataset):
    '''
    Training on COCOStuff dataset. [train2017.zip]

    Download the training set from https://github.com/nightrome/cocostuff
    '''

    def __init__(self, ann_file, data_prefix, pipeline, test_mode=False):
        super(COCOStuff_Full_Dataset, self).__init__(pipeline, test_mode)
        self.ann_file = str(ann_file)
        self.data_prefix = data_prefix
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        """Load annotations for dataset.

        Returns:
            list[dict]: Contain dataset annotations.
        """
        with open(self.ann_file, 'r') as f:
            img_infos = []
            for idx, line in enumerate(f):
                line = line.strip()
                _info = dict()
                img_path = line.split(' ')[0].split('/')[1]
                _info = dict(
                    gt_img_path=Path(
                        self.data_prefix).joinpath(img_path).as_posix(),
                    gt_img_idx=idx)
                img_infos.append(_info)

        return img_infos


@DATASETS.register_module()
class COCOStuff_Instance_Dataset(BaseDataset):
    '''
    Training on COCOStuff dataset. [train2017.zip]

    Download the training set from
        https://github.com/nightrome/cocostuff

    Make sure you've predicted all the images' bounding boxes using \
    MaskRCNN from detectron2

    It would be better if you can filter out the images which don't
    have any box.
    '''

    def __init__(self,
                 ann_file,
                 data_prefix,
                 npz_prefix,
                 pipeline,
                 test_mode=False):
        super(COCOStuff_Instance_Dataset, self).__init__(pipeline, test_mode)
        self.ann_file = str(ann_file)
        self.data_prefix = data_prefix
        self.npz_prefix = npz_prefix
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        """Load annotations for dataset.

        Returns:
            list[dict]: Contain dataset annotations.
        """
        with open(self.ann_file, 'r') as f:
            img_infos = []
            for idx, line in enumerate(f):
                line = line.strip()
                _info = dict()
                line_split = line.split(' ')
                img_path = line_split[0].split('/')[1]
                bbox_path = line_split[1].split('/')[1]
                _info = dict(
                    gt_img_path=Path(
                        self.data_prefix).joinpath(img_path).as_posix(),
                    gt_bbox_path=Path(
                        self.npz_prefix).joinpath(bbox_path).as_posix(),
                    gt_img_idx=idx)
                img_infos.append(_info)

        return img_infos


class COCOStuff_Fusion_Dataset(BaseDataset):
    '''
    Training on COCOStuff dataset. [train2017.zip]

    Download the training set from
        https://github.com/nightrome/cocostuff

    Make sure you've predicted all the images' bounding boxes
    using inference_bbox.py

    It would be better if you can filter out the images which
    don't have any box.
    '''

    def __init__(self, ann_file, data_prefix, pipeline, test_mode=False):
        super(COCOStuff_Fusion_Dataset, self).__init__(pipeline, test_mode)
        self.ann_file = str(ann_file)
        self.data_prefix = data_prefix
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
                    gt_bbox_path=Path(self.data_prefix).joinpath(
                        line_split[1]).as_posix(),
                    gt_img_idx=idx)
                img_infos.append(_info)

        return img_infos
