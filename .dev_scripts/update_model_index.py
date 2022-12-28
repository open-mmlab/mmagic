#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.

# This tool is used to update model-index.yml which is required by MIM, and
# will be automatically called as a pre-commit hook. The updating will be
# triggered if any change of model information (.md files in configs/) has been
# detected before a commit.

import glob
import os
import posixpath as osp  # Even on windows, use posixpath
import re
import sys
import warnings
from functools import reduce

import mmengine

MMEditing_ROOT = osp.dirname(osp.dirname(__file__))

all_training_data = [
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


def get_task_name(md_file):
    """Get task name from README.md".

    Args:
        md_file (str): Path to .md file.

    Returns:
        Str: Task name.
    """
    layers = re.split(r'[\\/]', md_file)
    for i in range(len(layers) - 1):
        if layers[i] == 'configs':
            return layers[i + 1].capitalize()
    return 'Unknown'


def generate_unique_name(md_file):
    """Search config files and return the unique name of them.

    Args:
        md_file (str): Path to .md file.
    Returns:
        dict: dict of unique name for each config file.
    """
    files = os.listdir(osp.dirname(md_file))
    config_files = [f[:-3] for f in files if f[-3:] == '.py']
    config_files.sort()
    config_files.sort(key=lambda x: len(x))
    split_names = [f.split('_') for f in config_files]
    config_sets = [set(f.split('_')) for f in config_files]
    common_set = reduce(lambda x, y: x & y, config_sets)
    unique_lists = [[n for n in name if n not in common_set]
                    for name in split_names]

    unique_dict = dict()
    name_list = []
    for i, f in enumerate(config_files):
        base = split_names[i][0]
        unique_dict[f] = base
        if len(unique_lists[i]) > 0:
            for unique in unique_lists[i]:
                candidate_name = f'{base}_{unique}'
                if candidate_name not in name_list and base != unique:
                    unique_dict[f] = candidate_name
                    name_list.append(candidate_name)
                    break
    return unique_dict


def parse_md(md_file):
    """Parse .md file and convert it to a .yml file which can be used for MIM.

    Args:
        md_file (str): Path to .md file.
    Returns:
        Bool: If the target YAML file is different from the original.
    """
    # See https://github.com/open-mmlab/mmediting/pull/798 for these comments
    # unique_dict = generate_unique_name(md_file)

    collection_name = osp.splitext(osp.basename(md_file))[0]
    readme = osp.relpath(md_file, MMEditing_ROOT)
    readme = readme.replace('\\', '/')  # for windows
    last_gpu_info = None
    collection = dict(
        Name=collection_name,
        Metadata={'Architecture': []},
        README=readme,
        Paper=[])
    models = []
    # force utf-8 instead of system defined
    with open(md_file, 'r', encoding='utf-8') as md:
        lines = md.readlines()
        i = 0
        name = lines[0][2:]
        name = name.split('(', 1)[0].strip()
        collection['Metadata']['Architecture'].append(name)
        collection['Name'] = name
        collection_name = name
        is_liif = collection_name.upper() == 'LIIF'
        task_line = lines[4]
        task = task_line.strip().split(':')[-1].strip()
        while i < len(lines):
            # parse reference
            if lines[i].startswith('> ['):
                url = re.match(r'> \[.*]\((.*)\)', lines[i])
                url = url.groups()[0]
                collection['Paper'].append(url)
                i += 1

            # parse table
            elif (lines[i][0] == '|') and (i + 1 < len(lines)) and (
                    lines[i + 1][:3] == '| :' or lines[i + 1][:2] == '|:'
                    or lines[i + 1][:2] == '|-') and (
                        'SKIP THIS TABLE' not in lines[i - 2]  # for aot-gan
                    ):
                cols = [col.strip() for col in lines[i].split('|')][1:-1]
                # if 'cain' in name.lower():
                #     import ipdb
                #     ipdb.set_trace()
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

                checkpoint_idx = cols.index('Download')
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

                    if line[checkpoint_idx].find('](') >= 0:
                        if line[checkpoint_idx].find('model](') >= 0:
                            left = line[checkpoint_idx].index('model](') + 7
                        else:
                            left = line[checkpoint_idx].index('ckpt](') + 6
                        right = line[checkpoint_idx].index(')', left)
                        checkpoint = line[checkpoint_idx][left:right]
                    else:
                        checkpoint = ''

                    name_key = osp.splitext(osp.basename(config))[0]
                    model_name = name_key
                    # See https://github.com/open-mmlab/mmediting/pull/798
                    # for these comments
                    # if name_key in unique_dict:
                    #     model_name = unique_dict[name_key]
                    # else:
                    #     model_name = name_key
                    #     warnings.warn(
                    #         f'Config file of {model_name} is not found,'
                    #         'please check it again.')

                    # find dataset in config file
                    dataset = 'Others'
                    config_low = config.lower()
                    for d in all_training_data:
                        if d in config_low:
                            dataset = d.upper()
                            break
                    metadata = {'Training Data': dataset}
                    if flops_idx != -1:
                        metadata['FLOPs'] = float(line[flops_idx])
                    if params_idx != -1:
                        metadata['Parameters'] = float(line[params_idx])
                    if gpu_idx != -1:
                        metadata['GPUs'] = line[gpu_idx].strip()
                        if '△' in metadata['GPUs']:
                            metadata['GPUs'] = last_gpu_info
                        else:
                            last_gpu_info = metadata['GPUs']

                    metrics = {}

                    for key in used_metrics:
                        # handle scale for LIIF model
                        if key.upper() == 'SCALE':
                            # remove 'x' in scale
                            scale = line[used_metrics[key]].strip()[1:]

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

                    if is_liif:
                        new_metrics = dict()
                        for k, v in metrics.items():
                            dataset, metric_name = k.split(' ')
                            new_metrics[f'{dataset}x{scale} {metric_name}'] = v
                        metrics = new_metrics

                    if is_liif and models and models[-1]['Name'] == model_name:
                        models[-1]['Results'][0]['Metrics'].update(metrics)
                    else:
                        model = {
                            'Name':
                            model_name,
                            'In Collection':
                            collection_name,
                            'Config':
                            config,
                            'Metadata':
                            metadata,
                            'Results': [{
                                'Task': task,
                                'Dataset': dataset,
                                'Metrics': metrics
                            }],
                            'Weights':
                            checkpoint
                        }
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
    # pbar = tqdm.tqdm(range(len(file_list)), initial=0, dynamic_ncols=True)
    for fn in file_list:
        file_modified |= parse_md(fn)
        # pbar.update(1)
        # pbar.set_description(f'processing {fn}')

    file_modified |= update_model_index()

    sys.exit(1 if file_modified else 0)
