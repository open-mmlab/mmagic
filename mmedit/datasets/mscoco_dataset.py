# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Union

import mmengine
from mmengine import FileClient

from mmedit.registry import DATASETS

import os

from .basic_conditional_dataset import BasicConditionalDataset
import random

IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF')


@DATASETS.register_module()
@DATASETS.register_module("MSCOCO")
class MSCoCoDataset(BasicConditionalDataset):

    METAINFO = dict(dataset_type='text_image_dataset', task_name='editing')

    def __init__(self,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_root: str = '',
                 drop_caption_rate=0.0,
                 phase='train',
                 year=2014,
                 data_prefix: Union[str, dict] = '',
                 extensions: Sequence[str] = ('.jpg', '.jpeg', '.png', '.ppm',
                                              '.bmp', '.pgm', '.tif'),
                 lazy_init: bool = False,
                 classes: Union[str, Sequence[str], None] = None,
                 **kwargs):
        ann_file = os.path.join("annotations", "captions_" + phase +
                                f"{year}.json") if ann_file == '' else ann_file
        self.image_prename = "COCO_" + phase + f"{year}_"
        self.phase = phase
        self.drop_rate = drop_caption_rate
        self.year = year

        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            extensions=extensions,
            lazy_init=lazy_init,
            classes=classes,
            **kwargs)

    def load_data_list(self):
        """Load image paths and gt_labels."""
        if self.img_prefix:
            file_client = FileClient.infer_client(uri=self.img_prefix)
        json_file = mmengine.fileio.io.load(self.ann_file)

        def add_prefix(filename, prefix=''):
            if not prefix:
                return filename
            else:
                return file_client.join_path(prefix, filename)

        data_list = []
        for item in json_file['annotations']:
            image_name = self.image_prename + str(
                item['image_id']).zfill(12) + '.jpg'
            img_path = add_prefix(
                os.path.join(self.phase + str(self.year), image_name),
                self.img_prefix)
            caption = item['caption'].lower()
            info = {
                'img_path':
                img_path,
                'caption':
                caption if (self.phase != 'train' or self.drop_rate < 1e-6
                            or random.random() >= self.drop_rate) else ''
            }
            data_list.append(info)
        return data_list
