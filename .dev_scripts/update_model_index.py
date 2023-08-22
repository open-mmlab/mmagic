#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""This tool is used to update model-index.yml which is required by MIM, and
will be automatically called as a pre-commit hook. The updating will be
triggered if any change of model information (.md files in configs/) has been
detected before a commit.

The design of the metafile follows /wikcnz2ksDIyZiFM8iAMUoTmNHg.
It is forbidden to set value as `NULL`, `None` or empty list and str.

`Collection`: indicates a algorithm (which may
    contains multiple configs), e.g. Mask R-CNN
    - `Name` (str): required
    - `README` (str): optional
    - `Paper` (dict): optional
        - `URL` (str): required
        - `Title` (str): required
    - `Task` (List[str]): optional
    - `Year` (int): optional
    - `Metadata` (dict): optional
        - `Architecture` (List[str]): optional
        - `Training Data` (Union[str, List[str]]): optional
        - `Epochs` (int): optional
        - `Batch Size` (int): optional
        - `Training Techniques` (List[str]): optional
        - `Training Resources` (str): optional
        - `FLOPs` (Union[int, float]): optional
        - `Parameters` (int): optional
        - `Training Time` (Union[int, float]): optional
        - `Train time (s/iter)` (Union[int, float]): optional
        - `Training Memory (GB)` (float): optional
    - `Weights` (Union[str, List[str]]): optional

`Model`: indicates a specific config
    - `Name` (str): required, globally unique
    - `In Collection` (str): required
    - `Config` (str): required
    - `Results` (List[dict]): required
        - `Task` (str): required
        - `Dataset` (str): required
        - `Metrics` (dict): required
    - `Weights` (str): optional
    - `Metadata` (dict): optional
        - `Architecture` (List[str]): optional
        - `Training Resources` (str): optional
        - `Training Data` (Union[str, List[str]]): optional
        - `Epochs` (int): optional
        - `Batch Size` (int): optional
        - `Training Techniques` (List[str]): optional
        - `FLOPs` (Union[int, float]): optional
        - `Parameters` (int): optional
        - `Training Time` (Union[int, float]): optional
        - `Train time (s/iter)` (Union[int, float]): optional
        - `Training Memory (GB)` (float): optional
        - `inference time (ms/im)` (List[dict]): optional
            - `value` (float): required
            - `hardware` (str): required
            - `backend` (str): required
            - `batch size` (str): required
            - `mode` (str): required, e.g., FP32, FP16, INT8, etc.
            - `resolution` (Tuple(int, int)): required
    - `Training Log` (str): optional
    - `README` (str): optional
    - `Paper` (dict): optional
        - `URL` (str): required
        - `Title` (str): required
    - `Converted From` (dict): optional
        - `Weights` (str): required
        - `Code` (str): required
    - `Code` (dict): optional
        - `URL` (str): required
        - `Version` (str): required
    - `Image` (str): optional

The README.md file in MMagic follows following convention,
    - `Model`: [name_in_the_paper](path_to_config)
    - `Download`: [model](url_to_pre_trained_weights) | [log](url_to_log)
"""

import glob
import posixpath as osp  # Even on windows, use posixpath
import re
import sys

import tqdm
from modelindex.models.Collection import Collection
from modelindex.models.Metadata import Metadata
from modelindex.models.Model import Model
from modelindex.models.Result import Result
from utils import (collate_metrics, dump_yaml_and_check_difference,
                   found_table, modelindex_to_dict)

MMagic_ROOT = osp.dirname(osp.dirname(__file__))
KEYWORDS = [
    'Model',
    'Dataset',
    'Download',
]


def parse_md(md_file):
    """Parse .md file and convert it to a .yml file which can be used for MIM.

    Args:
        md_file (str): Path to .md file.
    Returns:
        Bool: If the target YAML file is different from the original.
    """
    readme = osp.relpath(md_file, MMagic_ROOT)
    readme = readme.replace('\\', '/')  # for windows
    collection = Collection()
    collection_meta = Metadata()

    # force utf-8 instead of system defined
    with open(md_file, 'r', encoding='utf-8') as md:
        lines = md.readlines()

    # parse information for collection
    name = lines[0][2:]
    name, year = name.split('(', 1)
    year = int(re.sub('[^0-9]', '', year))
    collection_name = name.strip()
    task_line = lines[4]
    model_task = task_line.strip().split(':')[-1].strip()
    model_task_list = model_task.lower().split(', ')

    collection.name = collection_name
    collection.readme = readme
    collection.data['Year'] = year
    collection.data['Task'] = model_task_list
    collection_meta.architecture = [collection_name]

    i = 0
    model_list = {}
    while i < len(lines):

        # parse reference
        if lines[i].startswith('> ['):
            title, url = re.match(r'> \[(.*)]\((.*)\)', lines[i]).groups()
            collection.paper = dict(URL=url, Title=title)
            i += 1

        # parse table
        elif found_table(lines, i):
            cols = [col.strip() for col in lines[i].split('|')][1:-1]

            # check required field for Model
            try:
                model_idx = cols.index('Model')
                dataset_idx = cols.index('Dataset')
                download_idx = cols.index('Download')
                used_metrics = collate_metrics(cols)
                if 'Task' in cols:
                    task_idx = cols.index('Task')
                else:
                    task_idx = None
            except Exception:
                raise ValueError(
                    f'required fields: Model, Dataset, Download '
                    f'are not included in line {i+1} of {md_file}')

            j = i + 2
            # parse table for valid fields
            while j < len(lines) and lines[j][0] == '|':
                line = lines[j].split('|')[1:-1]

                # name, in_collection, config of Model
                assert line[model_idx].find(
                    '](') >= 0, f'invalid {model_idx} in {line} for {md_file}.'
                left = line[model_idx].index('](') + 2
                right = line[model_idx].index(')', left)
                config = line[model_idx][left:right].strip('./')
                model_name = osp.splitext(osp.basename(config))[0]
                if model_name in model_list:
                    model = model_list[model_name]
                else:
                    model = Model(
                        name=model_name,
                        in_collection=collection_name,
                        config=osp.join(
                            osp.dirname(md_file), model_name + '.py'))
                    model_list[model_name] = model

                # find results in each row
                dataset = line[dataset_idx].replace(' ', '')

                metrics = {}
                for metric_name, idx in used_metrics.items():
                    value = line[idx]
                    value = value.replace('*', '')
                    if '/' not in value:
                        try:
                            value = value.split('(')[0]
                            metrics[metric_name] = float(value)
                        except ValueError:
                            value = value.replace(' ', '')
                    else:
                        try:
                            PSNR, SSIM = [float(d) for d in value.split('/')]
                            metrics[metric_name] = dict(PSNR=PSNR, SSIM=SSIM)
                        except ValueError:
                            pass

                task = model_task if task_idx is None else line[
                    task_idx].strip()
                assert ',' not in task, (
                    f'Find "," in "task" field of "{md_file}" (line {j}). '
                    'Please check your readme carefully.')
                assert task.lower() in model_task_list, (
                    f'Task "{task}" not in "{model_task_list}" in "{md_file}" '
                    f'(line {j}). Please check your readme carefully.')

                result = Result(task=task, dataset=dataset, metrics=metrics)
                if model.results is None:
                    model.results = result
                else:
                    model.results.data.append(result)

                # check weights
                if line[download_idx].find('](') >= 0:
                    if line[download_idx].find('model](') >= 0:
                        left = line[download_idx].index('model](') + 7
                    else:
                        left = line[download_idx].index('ckpt](') + 6
                    right = line[download_idx].index(')', left)
                    model.weights = line[download_idx][left:right]
                j += 1
            i = j

        else:
            i += 1

    collection = modelindex_to_dict(collection)
    models = [modelindex_to_dict(m) for n, m in model_list.items()]
    assert len(models) > 0, f"'no model is found in {md_file}'"
    result = {'Collections': [collection], 'Models': models}
    yml_file = md_file.replace('README.md', 'metafile.yml')
    is_different = dump_yaml_and_check_difference(result, yml_file)

    return is_different


def update_model_index():
    """Update model-index.yml according to model .md files.

    Returns:
        Bool: If the updated model-index.yml is different from the original.
    """
    configs_dir = osp.join(MMagic_ROOT, 'configs')
    yml_files = glob.glob(osp.join(configs_dir, '**', '*.yml'), recursive=True)
    yml_files.sort()

    model_index = {
        'Import': [
            osp.relpath(yml_file, MMagic_ROOT).replace(
                '\\', '/')  # force using / as path separators
            for yml_file in yml_files
        ]
    }
    model_index_file = osp.join(MMagic_ROOT, 'model-index.yml')
    is_different = dump_yaml_and_check_difference(model_index,
                                                  model_index_file)

    return is_different


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        configs_root = osp.join(MMagic_ROOT, 'configs')
        file_list = glob.glob(
            osp.join(configs_root, '**', '*README.md'), recursive=True)
        file_list.sort()
    else:
        file_list = [
            fn for fn in sys.argv[1:] if osp.basename(fn) == 'README.md'
        ]

    if not file_list:
        sys.exit(0)

    file_modified = False
    file_list = file_list
    pbar = tqdm.tqdm(range(len(file_list)), initial=0, dynamic_ncols=True)
    for fn in file_list:
        file_modified |= parse_md(fn)
        pbar.update(1)
        pbar.set_description(f'processing {fn}')

    file_modified |= update_model_index()

    sys.exit(1 if file_modified else 0)
