# Copyright (c) OpenMMLab. All rights reserved.
import os
import random
from typing import Any, Callable, List, Union

from mmengine import FileClient
from mmengine.dataset import BaseDataset
from PIL import Image
from torchvision import transforms

from mmagic.registry import DATASETS

imagenet_templates_smallest = [
    'a photo of a {}',
]

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
    'an illustration of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'an illustration of a clean {}',
    'an illustration of a dirty {}',
    'a dark photo of the {}',
    'an illustration of my {}',
    'an illustration of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'an illustration of the {}',
    'a good photo of the {}',
    'an illustration of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'an illustration of the clean {}',
    'a rendition of a {}',
    'an illustration of a nice {}',
    'a good photo of a {}',
    'an illustration of the nice {}',
    'an illustration of the small {}',
    'an illustration of the weird {}',
    'an illustration of the large {}',
    'an illustration of a cool {}',
    'an illustration of a small {}',
    'a depiction of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'a depiction of a clean {}',
    'a depiction of a dirty {}',
    'a dark photo of the {}',
    'a depiction of my {}',
    'a depiction of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'a depiction of the {}',
    'a good photo of the {}',
    'a depiction of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'a depiction of the clean {}',
    'a rendition of a {}',
    'a depiction of a nice {}',
    'a good photo of a {}',
    'a depiction of the nice {}',
    'a depiction of the small {}',
    'a depiction of the weird {}',
    'a depiction of the large {}',
    'a depiction of a cool {}',
    'a depiction of a small {}',
]

imagenet_dual_templates_small = [
    'a photo of a {} with {}',
    'a rendering of a {} with {}',
    'a cropped photo of the {} with {}',
    'the photo of a {} with {}',
    'a photo of a clean {} with {}',
    'a photo of a dirty {} with {}',
    'a dark photo of the {} with {}',
    'a photo of my {} with {}',
    'a photo of the cool {} with {}',
    'a close-up photo of a {} with {}',
    'a bright photo of the {} with {}',
    'a cropped photo of a {} with {}',
    'a photo of the {} with {}',
    'a good photo of the {} with {}',
    'a photo of one {} with {}',
    'a close-up photo of the {} with {}',
    'a rendition of the {} with {}',
    'a photo of the clean {} with {}',
    'a rendition of a {} with {}',
    'a photo of a nice {} with {}',
    'a good photo of a {} with {}',
    'a photo of the nice {} with {}',
    'a photo of the small {} with {}',
    'a photo of the weird {} with {}',
    'a photo of the large {} with {}',
    'a photo of a cool {} with {}',
    'a photo of a small {} with {}',
]


@DATASETS.register_module()
class ViCoDataset(BaseDataset):
    """Dataset for DreamBooth.

    Args:
        data_root (str): Path to the data root.
        concept_dir (str): Path to the concept images.
        prompt (str): Prompt of the concept.
        pipeline (list[dict | callable]): A sequence of data transforms.
    """

    def __init__(self,
                 data_root: str,
                 concept_dir: str,
                 placeholder: str,
                 pipeline: List[Union[dict, Callable]] = []):

        data_prefix = dict(img_path=concept_dir)
        self.num_images = len(os.listdir(os.path.join(data_root, concept_dir)))
        self.placeholder = placeholder
        self.template = imagenet_templates_small

        super().__init__(
            data_root=data_root, data_prefix=data_prefix, pipeline=pipeline)

    def load_data_list(self) -> list:
        """Load data list from concept_dir and class_dir."""
        data_list = []

        img_dir = self.data_prefix['img_path']
        file_client = FileClient.infer_client(uri=img_dir)
        img_dir = os.path.abspath(img_dir)

        data_names = file_client.list_dir_or_file(img_dir, list_dir=False)
        for data_name in data_names:
            data_info = dict(
                img_path=file_client.join_path(img_dir, data_name))
            data_list.append(data_info)

        return data_list

    def prepare_data(self, idx) -> Any:
        data_info = self.get_data_info(idx)

        numbers = list(range(self.num_images))
        if len(numbers) > 1:
            numbers.remove(idx % self.num_images)
        img_dir = self.data_prefix['img_path']
        file_client = FileClient.infer_client(uri=img_dir)
        img_dir = os.path.abspath(img_dir)
        data_names = list(
            file_client.list_dir_or_file(img_dir, list_dir=False))
        image_ref_path = file_client.join_path(
            img_dir, data_names[random.choice(numbers)])
        data_info['img_ref_path'] = image_ref_path

        selected_template = random.choice(self.template)
        prompt = selected_template.format(self.placeholder)
        data_info['prompt'] = prompt
        return self.pipeline(data_info)
