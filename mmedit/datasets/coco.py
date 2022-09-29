# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Callable, Optional, Union
from pathlib import Path

from mmengine.dataset import BaseDataset
from mmengine.fileio import load

from mmedit.registry import DATASETS


@DATASETS.register_module()
class CocoDataset:
    """Dataset for COCO."""

    METAINFO = {
        'CLASSES':
            ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
             'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
             'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
             'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
             'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
             'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
             'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
             'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
             'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
             'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
             'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
             'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
             'scissors', 'teddy bear', 'hair drier', 'toothbrush'),
        # PALETTE is a list of color tuples, which is used for visualization.
        'PALETTE':
            [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
             (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
             (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
             (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
             (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
             (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
             (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
             (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),
             (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174),
             (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54),
             (207, 138, 255), (151, 0, 95), (9, 80, 61), (84, 105, 51),
             (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
             (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
             (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161),
             (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120),
             (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133),
             (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62),
             (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45),
             (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
             (246, 0, 122), (191, 162, 208)]
    }

    METAINFO = dict(dataset_type='colorization_dataset', task_name='colorization')

    def __init__(
            self,
            ann_file: str,
            data_prefix
    ):

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
