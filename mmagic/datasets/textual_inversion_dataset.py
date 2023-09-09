# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from random import choice
from typing import Callable, List, Union

from mmengine import FileClient
from mmengine.dataset import BaseDataset

from mmagic.registry import DATASETS


@DATASETS.register_module()
class TextualInversionDataset(BaseDataset):
    """Dataset for Textual Inversion and ViCo.

    Args:
        data_root (str): Path to the data root.
        concept_dir (str): Path to the concept images.
        placeholder (str): A string to denote the concept.
        template (list[str]): A list of strings like 'A photo of {}'.
        with_image_reference (bool): Is used for vico training.
        pipeline (list[dict | callable]): A sequence of data transforms.
    """

    def __init__(
            self,
            data_root: str,
            concept_dir: str,
            placeholder: str,
            template: str,
            # used for vico training
            with_image_reference: bool = False,
            pipeline: List[Union[dict, Callable]] = []):

        data_prefix = dict(img_path=concept_dir)

        self.placeholder = placeholder
        if osp.exists(osp.join(data_root, concept_dir)):
            self.num_images = len(os.listdir(osp.join(data_root, concept_dir)))
        if osp.exists(template):
            with open(template, 'r') as file:
                self.template = file.readlines()
        self.with_image_reference = with_image_reference

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

        if self.with_image_reference:
            numbers = list(range(self.num_images))
            if len(numbers) > 1:
                numbers.remove(idx % self.num_images)
            img_dir = self.data_prefix['img_path']
            file_client = FileClient.infer_client(uri=img_dir)
            img_dir = osp.abspath(img_dir)
            data_names = list(
                file_client.list_dir_or_file(img_dir, list_dir=False))
            image_ref_path = file_client.join_path(img_dir,
                                                   data_names[choice(numbers)])
            data_info['img_ref_path'] = image_ref_path

        # load random template
        selected_template = choice(self.template)
        prompt = selected_template.format(self.placeholder)
        data_info['prompt'] = prompt
        return self.pipeline(data_info)
