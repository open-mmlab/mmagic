# Copyright (c) OpenMMLab. All rights reserved.
import os
import random
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
from mmengine.dataset import BaseDataset

from mmagic.registry import DATASETS


@DATASETS.register_module()
class HuggingFaceDataset(BaseDataset):
    """Huggingface Dataset for DreamBooth.

    Args:
        dataset (str): Dataset name for Huggingface datasets.
        image_column (str): Image column name. Defaults to 'image'.
        caption_column (str): Caption column name. Defaults to 'text'.
        csv (str): Caption csv file name when loading local folder.
            Defaults to 'metadata.csv'.
        cache_dir (str, optional): The directory where the downloaded datasets
            will be stored.Defaults to None.
        pipeline (list[dict | callable]): A sequence of data transforms.
    """

    def __init__(self,
                 dataset: str,
                 image_column: str = 'image',
                 caption_column: str = 'text',
                 csv: str = 'metadata.csv',
                 cache_dir: Optional[str] = None,
                 pipeline: List[Union[dict, Callable]] = []):

        self.dataset = dataset
        self.image_column = image_column
        self.caption_column = caption_column
        self.csv = csv
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

        if Path(self.dataset).exists():
            # load local folder
            data_file = os.path.join(self.dataset, self.csv)
            dataset = load_dataset(
                'csv', data_files=data_file, cache_dir=self.cache_dir)['train']
        else:
            # load huggingface online
            dataset = load_dataset(
                self.dataset, cache_dir=self.cache_dir)['train']

        for i in range(len(dataset)):
            caption = dataset[i][self.caption_column]
            if isinstance(caption, str):
                pass
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                caption = random.choice(caption)
            else:
                raise ValueError(
                    f'Caption column `{self.caption_column}` should contain'
                    ' either strings or lists of strings.')

            img = dataset[i][self.image_column]
            if type(img) == str:
                img = os.path.join(self.dataset, img)

            data_info = dict(img=img, prompt=caption)
            data_list.append(data_info)

        return data_list
