# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
from typing import Callable, List, Union

from mmengine.dataset import BaseDataset

from mmagic.registry import DATASETS


@DATASETS.register_module()
class ControlNetDataset(BaseDataset):
    """Demo dataset to test ControlNet. Modified from https://github.com/lllyas
    viel/ControlNet/blob/16ea3b5379c1e78a4bc8e3fc9cae8d65c42511b1/tutorial_data
    set.py  # noqa.

    You can download the demo data from https://huggingface.co/lllyasviel/ControlNet/blob/main/training/fill50k.zip  # noqa
    and then unzip the file to the ``data`` folder.

    Args:
        ann_file (str): Path to the annotation file. Defaults
            to 'prompt.json' as ControlNet's default.
        data_root (str): Path to the data root. Defaults to './data/fill50k'.
        pipeline (list[dict | callable]): A sequence of data transforms.
    """

    def __init__(self,
                 ann_file: str = 'prompt.json',
                 data_root: str = './data/fill50k',
                 control_key='source',
                 image_key='target',
                 pipeline: List[Union[dict, Callable]] = []):
        self.control_key = control_key
        self.image_key = image_key
        super().__init__(
            ann_file=ann_file, data_root=data_root, pipeline=pipeline)

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            list[dict]: A list of annotation.
        """
        data_list = []
        with open(self.ann_file, 'rt') as file:
            anno_list = file.readlines()

        for anno in anno_list:
            anno = json.loads(anno)
            # source = anno['source']
            # target = anno['target']
            source = anno[self.control_key]
            target = anno[self.image_key]
            prompt = anno['prompt']

            source = os.path.join(self.data_root, source)
            target = os.path.join(self.data_root, target)

            data_list.append({
                'source_path': source,
                'target_path': target,
                'prompt': prompt
            })

        return data_list
