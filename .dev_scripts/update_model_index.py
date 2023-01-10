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
    - `Metadata` (dict): required
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
        - `inference time (ms/im)` (List[dict]): required
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
"""

import glob
import posixpath as osp  # Even on windows, use posixpath
import re
import sys
import warnings

import mmengine
from modelindex.Collection import Collection
from modelindex.Metadata import Metadata
from modelindex.models.Model import Model
from modelindex.models.Result import Result

MMEditing_ROOT = osp.dirname(osp.dirname(__file__))
TRAINING_DATA = [
    'div2k', 'celeba', 'places', 'comp1k', 'vimeo90k', 'reds', 'ffhq', 'cufed',
    'cat', 'facades', 'summer2winter', 'horse2zebra', 'maps', 'edges2shoes'
]


def dump_yaml_and_check_difference(obj, file):
    """Dump object to a yaml file, and check if the file content is different
    from the original.

    Args:
        obj (any): The python object to be dumped.
        file (str): YAML filename to dump the object to.
    Returns:
        Bool: If the target YAML file is different from the original.
    """

    str_dump = mmengine.dump(
        obj, None, file_format='yaml', sort_keys=True,
        line_break='\n')  # force use LF

    if osp.isfile(file):
        file_exists = True
        # print(f'    exist {file}')
        with open(file, 'r', encoding='utf-8') as f:
            str_orig = f.read()
    else:
        file_exists = False
        str_orig = None

    if file_exists and str_orig == str_dump:
        is_different = False
    else:
        is_different = True
        print(f'    update {file}')
        with open(file, 'w', encoding='utf-8') as f:
            f.write(str_dump)

    return is_different


def collate_metrics(keys):
    """Collect metrics from the first row of the table.

    Args:
        keys (List): Elements in the first row of the table.

    Returns:
        dict: A dict of metrics.
    """
    used_metrics = dict()
    for idx, key in enumerate(keys):
        if key in ['Method', 'Download', 'GPU Info']:
            continue
        used_metrics[key] = idx
    return used_metrics


def check_empty_value(obj):
    if isinstance(obj, str):
        assert obj != '', 'The value of the string shouldn\'t set as empty.'
    if isinstance(obj, list):
        assert len(obj) > 0, 'The list shouldn\'t be empty.'
        for v in obj:
            check_empty_value(v)
    if isinstance(obj, dict):
        assert len(list(obj.keys())) > 0, 'The dict shouldn\'t be empty.'
        for k, v in obj.items():
            check_empty_value(v)


def table_need_parse(line, line_plus_1, line_minus_2):
    return (line[0]
            == '|') and (line_plus_1[:3] == '| :' or line_plus_1[:2] == '|:'
                         or line_plus_1[:2] == '|-') and ('SKIP THIS TABLE'
                                                          not in line_minus_2)


def parse_md(md_file):
    """Parse .md file and convert it to a .yml file which can be used for MIM.

    Args:
        md_file (str): Path to .md file.
    Returns:
        Bool: If the target YAML file is different from the original.
    """

    readme = osp.relpath(md_file, MMEditing_ROOT)
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
    task = task_line.strip().split(':')[-1].strip()

    collection.name = collection_name
    collection.readme = readme
    collection['Year'] = year
    collection['Task'] = task.lower().split(', ')
    collection_meta.architecture = [collection_name]

    i = 0
    models = []
    last_gpu_info = None
    while i < len(lines):

        # parse reference
        if lines[i].startswith('> ['):
            title, url = re.match(r'> \[(.*)]\((.*)\)', lines[i]).groups()
            collection.paper = dict(URL=url, Title=title)
            i += 1

        # parse table
        elif i + 1 < len(lines) and table_need_parse(lines, lines[i + 1],
                                                     lines[i - 2]):
            cols = [col.strip() for col in lines[i].split('|')][1:-1]
            if 'Config' not in cols and 'Download' not in cols:
                warnings.warn("Lack 'Config' or 'Download' in"
                              f'line {i+1} in {md_file}')
                i += 1
                continue

            if 'Method' in cols:
                config_idx = cols.index('Method')
            elif 'Config' in cols:
                config_idx = cols.index('Config')
            else:
                print(cols)
                raise ValueError('Cannot find config Table.')

            # checkpoint_idx = cols.index('Download')
            try:
                flops_idx = cols.index('FLOPs')
            except ValueError:
                flops_idx = -1
            try:
                params_idx = cols.index('Params')
            except ValueError:
                params_idx = -1
            try:
                gpu_idx = cols.index('GPU Info')
            except ValueError:
                gpu_idx = -1
            used_metrics = collate_metrics(cols)

            j = i + 2
            while j < len(lines) and lines[j][0] == '|':
                line = lines[j].split('|')[1:-1]

                if line[config_idx].find('](') >= 0:
                    left = line[config_idx].index('](') + 2
                    right = line[config_idx].index(')', left)
                    config = line[config_idx][left:right].strip('./')
                    config = osp.join(
                        osp.dirname(md_file), osp.basename(config))
                elif line[config_idx].find('△') == -1:
                    j += 1
                    continue

                # if line[checkpoint_idx].find('](') >= 0:
                #     if line[checkpoint_idx].find('model](') >= 0:
                #         left = line[checkpoint_idx].index('model](') + 7
                #     else:
                #         left = line[checkpoint_idx].index('ckpt](') + 6
                #     right = line[checkpoint_idx].index(')', left)
                #     checkpoint = line[checkpoint_idx][left:right]

                name_key = osp.splitext(osp.basename(config))[0]
                model_name = name_key

                # find dataset in config file
                dataset = 'Others'
                config_low = config.lower()
                for d in TRAINING_DATA:
                    if d in config_low:
                        dataset = d.upper()
                        break
                metadata = {'Training Data': dataset}
                if flops_idx != -1:
                    metadata['FLOPs'] = float(line[flops_idx])
                if params_idx != -1:
                    metadata['Parameters'] = float(line[params_idx])
                if gpu_idx != -1:
                    metadata['Training Resources'] = line[gpu_idx].strip()
                    if '△' in metadata['Training Resources']:
                        metadata['Training Resources'] = last_gpu_info
                    else:
                        last_gpu_info = metadata['Training Resources']

                metrics = {}

                for key in used_metrics:
                    # handle scale for LIIF model
                    # if key.upper() == 'SCALE':
                    #     # remove 'x' in scale
                    #     scale = line[used_metrics[key]].strip()[1:]

                    metrics_data = line[used_metrics[key]]
                    metrics_data = metrics_data.replace('*', '')
                    if '/' not in metrics_data:
                        try:
                            metrics[key] = float(metrics_data)
                        except ValueError:
                            metrics_data = metrics_data.replace(' ', '')
                    else:
                        try:
                            metrics_data = [
                                float(d) for d in metrics_data.split('/')
                            ]
                            metrics[key] = dict(
                                PSNR=metrics_data[0], SSIM=metrics_data[1])
                        except ValueError:
                            pass

                # if is_liif:
                #     new_metrics = dict()
                #     for k, v in metrics.items():
                #         dataset, metric_name = k.split(' ')
                #         new_metrics[f'{dataset}x{scale} {metric_name}'] = v
                #     metrics = new_metrics

                # if is_liif and models and models[-1]['Name'] == model_name:
                #     models[-1]['Results'][0]['Metrics'].update(metrics)
                # else:
                model_meta = Metadata()
                result = Result(task=task, dataset=dataset, metrics=metrics)
                model = Model(
                    name=model_name,
                    in_collection=collection_name,
                    config=config,
                    metadata=model_meta,
                    results=[result])
                models.append(model)
                j += 1
            i = j

        else:
            i += 1

    if len(models) == 0:
        warnings.warn(f'no model is found in {md_file}')

    result = {'Collections': [collection], 'Models': models}
    yml_file = md_file.replace('README.md', 'metafile.yml')

    is_different = dump_yaml_and_check_difference(result, yml_file)
    return is_different


def update_model_index():
    """Update model-index.yml according to model .md files.

    Returns:
        Bool: If the updated model-index.yml is different from the original.
    """
    configs_dir = osp.join(MMEditing_ROOT, 'configs')
    yml_files = glob.glob(osp.join(configs_dir, '**', '*.yml'), recursive=True)
    yml_files.sort()

    model_index = {
        'Import': [
            osp.relpath(yml_file, MMEditing_ROOT).replace(
                '\\', '/')  # force using / as path separators
            for yml_file in yml_files
        ]
    }
    model_index_file = osp.join(MMEditing_ROOT, 'model-index.yml')
    is_different = dump_yaml_and_check_difference(model_index,
                                                  model_index_file)

    return is_different


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        configs_root = osp.join(MMEditing_ROOT, 'configs')
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
    import tqdm
    pbar = tqdm.tqdm(range(len(file_list)), initial=0, dynamic_ncols=True)
    for fn in file_list:
        file_modified |= parse_md(fn)
        pbar.update(1)
        pbar.set_description(f'processing {fn}')

    file_modified |= update_model_index()

    sys.exit(1 if file_modified else 0)
