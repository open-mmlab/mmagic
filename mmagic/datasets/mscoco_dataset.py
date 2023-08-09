# Copyright (c) OpenMMLab. All rights reserved.
import os
import random
from typing import Optional, Sequence, Union

import mmengine
from mmengine.fileio import get_file_backend

from mmagic.registry import DATASETS
from .basic_conditional_dataset import BasicConditionalDataset


@DATASETS.register_module()
@DATASETS.register_module('MSCOCO')
class MSCoCoDataset(BasicConditionalDataset):
    """MSCoCo 2014 dataset.

    Args:
        ann_file (str): Annotation file path. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to ''.
        drop_caption_rate (float, optional): Rate of dropping caption,
            used for training. Defaults to 0.0.
        phase (str, optional): Subdataset used for certain phase, can be set
            to `train`, `test` and `val`. Defaults to 'train'.
        year (int, optional): Version of CoCo dataset, can be set to 2014
            and 2017. Defaults to 2014.
        data_prefix (str | dict): Prefix for the data. Defaults to ''.
        extensions (Sequence[str]): A sequence of allowed extensions. Defaults
            to ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif').
        lazy_init (bool): Whether to load annotation during instantiation.
            In some cases, such as visualization, only the meta information of
            the dataset is needed, which is not necessary to load annotation
            file. ``Basedataset`` can skip load annotations to save time by set
            ``lazy_init=False``. Defaults to False.
        caption_style (str): If you want to add a style description for each
            caption, you can set caption_style to your style prompt. For
            example, 'realistic style'. Defaults to empty str.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """
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
                 caption_style: str = '',
                 **kwargs):
        ann_file = os.path.join('annotations', 'captions_' + phase +
                                f'{year}.json') if ann_file == '' else ann_file
        self.year = year
        assert self.year == 2014 or self.year == 2017, \
            'Caption is only supported in 2014 or 2017.'
        self.image_prename = ''
        if self.year == 2014:
            self.image_prename = 'COCO_' + phase + f'{year}_'
        self.phase = phase
        self.drop_rate = drop_caption_rate
        self.caption_style = caption_style

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
            file_backend = get_file_backend(uri=self.img_prefix)
        json_file = mmengine.fileio.io.load(self.ann_file)

        def add_prefix(filename, prefix=''):
            if not prefix:
                return filename
            else:
                return file_backend.join_path(prefix, filename)

        data_list = []
        for item in json_file['annotations']:
            image_name = self.image_prename + str(
                item['image_id']).zfill(12) + '.jpg'
            img_path = add_prefix(
                os.path.join(self.phase + str(self.year), image_name),
                self.img_prefix)
            caption = item['caption'].lower()
            if self.caption_style != '':
                caption = caption + ' ' + self.caption_style
            info = {
                'img_path':
                img_path,
                'gt_prompt':
                caption if (self.phase != 'train' or self.drop_rate < 1e-6
                            or random.random() >= self.drop_rate) else ''
            }
            data_list.append(info)
        return data_list
