# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, List, Optional, Union

from mmengine.dataset import BaseDataset

from mmagic.registry import DATASETS


@DATASETS.register_module()
class HuggingFaceDreamBoothDataset(BaseDataset):
    """Huggingface Dataset for DreamBooth.

    Args:
        dataset (str): Dataset name for Huggingface datasets.
        prompt (str): Prompt of the concept.
        image_column (str): Image column name. Defaults to 'image'.
        dataset_sub_dir (str, optional): Dataset sub directory name.
        cache_dir (str, optional): The directory where the downloaded datasets
            will be stored.Defaults to None.
        pipeline (list[dict | callable]): A sequence of data transforms.
    """

    def __init__(self,
                 dataset: str,
                 prompt: str,
                 image_column: str = 'image',
                 dataset_sub_dir: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 pipeline: List[Union[dict, Callable]] = []):

        self.dataset = dataset
        self.prompt = prompt
        self.image_column = image_column
        self.dataset_sub_dir = dataset_sub_dir
        self.cache_dir = cache_dir

        super().__init__(pipeline=pipeline)

    def load_data_list(self) -> list:
        """Load data list from concept_dir and class_dir."""
        try:
            from datasets import load_dataset
        except BaseException:
            raise ImportError(
                'HuggingFaceDreamBoothDataset requires datasets, please '
                'install it by `pip install datasets`.')

        data_list = []

        if self.dataset_sub_dir is not None:
            dataset = load_dataset(
                self.dataset, self.dataset_sub_dir,
                cache_dir=self.cache_dir)['train']
        else:
            dataset = load_dataset(
                self.dataset, cache_dir=self.cache_dir)['train']

        for i in range(len(dataset)):
            data_info = dict(
                img=dataset[i][self.image_column], prompt=self.prompt)
            data_list.append(data_info)

        return data_list
