# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from random import choice
from typing import Callable, List, Union

from mmengine import FileClient
from mmengine.dataset import BaseDataset

from mmagic.registry import DATASETS

imagenet_templates_small = [
    'a photo of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'a photo of a clean {}',
    'a photo of a dirty {}',
    'a dark photo of the {}',
    'a photo of my {}',
    'a photo of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'a photo of the {}',
    'a good photo of the {}',
    'a photo of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'a photo of the clean {}',
    'a rendition of a {}',
    'a photo of a nice {}',
    'a good photo of a {}',
    'a photo of the nice {}',
    'a photo of the small {}',
    'a photo of the weird {}',
    'a photo of the large {}',
    'a photo of a cool {}',
    'a photo of a small {}',
]

imagenet_style_templates_small = [
    'a painting in the style of {}',
    'a rendering in the style of {}',
    'a cropped painting in the style of {}',
    'the painting in the style of {}',
    'a clean painting in the style of {}',
    'a dirty painting in the style of {}',
    'a dark painting in the style of {}',
    'a picture in the style of {}',
    'a cool painting in the style of {}',
    'a close-up painting in the style of {}',
    'a bright painting in the style of {}',
    'a cropped painting in the style of {}',
    'a good painting in the style of {}',
    'a close-up painting in the style of {}',
    'a rendition in the style of {}',
    'a nice painting in the style of {}',
    'a small painting in the style of {}',
    'a weird painting in the style of {}',
    'a large painting in the style of {}',
]


@DATASETS.register_module()
class TextualInversionDataset(BaseDataset):
    """Dataset for DreamBooth.

    Args:
        data_root (str): Path to the data root.
        concept_dir (str): Path to the concept images.
        is_style (bool)
        prompt (str): Prompt of the concept.
        pipeline (list[dict | callable]): A sequence of data transforms.
    """

    def __init__(self,
                 data_root: str,
                 concept_dir: str,
                 placeholder: str,
                 is_style: bool = False,
                 pipeline: List[Union[dict, Callable]] = []):

        data_prefix = dict(img_path=concept_dir)

        self.placeholder = placeholder
        if is_style:
            self.template = imagenet_style_templates_small
        else:
            self.template = imagenet_templates_small

        super().__init__(
            data_root=data_root, data_prefix=data_prefix, pipeline=pipeline)

    def load_data_list(self) -> list:
        """Load data list from concept_dir and class_dir."""
        data_list = []

        img_dir = self.data_prefix['img_path']
        file_client = FileClient.infer_client(uri=img_dir)
        img_dir = osp.abspath(img_dir)

        for data_name in file_client.list_dir_or_file(img_dir, list_dir=False):
            data_info = dict(
                img_path=file_client.join_path(img_dir, data_name))
            data_list.append(data_info)
        return data_list

    def prepare_data(self, idx):
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        data_info = self.get_data_info(idx)
        # load random template
        selected_template = choice(self.template)
        prompt = selected_template.format(self.placeholder)
        data_info['prompt'] = prompt
        return self.pipeline(data_info)
