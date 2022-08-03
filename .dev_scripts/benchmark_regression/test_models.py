#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.

# This tool is used to update README.md and README_zh-CN.md in configs

import datetime
import glob
import os
import platform
import posixpath as osp  # Even on windows, use posixpath
import sys
import threading

import yaml

MMEditing_ROOT = osp.dirname(osp.dirname(osp.dirname(__file__)))
DOWNLOAD_DIR = osp.join(MMEditing_ROOT, 'work_dirs', 'download')
LOG_DIR = osp.join(
    MMEditing_ROOT, 'work_dirs',
    'benchmark_test_' + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
PARTITION = 'mm_lol'

sem = threading.Semaphore(8)  # The maximum number of restricted threads


def filter(info):
    """Filter the models you want to test.

    Args:
        info (dict): info of model.

    Returns:
        Bool: If this model should be tested.
    """

    # return 'global_local' in info['Config']
    return True


def find_available_port():
    """Find an available port.
    """

    port = 65535
    while True:
        if platform.system() == 'Windows':
            port_is_occupied = os.popen('netstat -an | findstr :' +
                                        str(port)).readlines()
        else:
            port_is_occupied = os.popen('netstat -antu | grep :' +
                                        str(port)).readlines()
        if not port_is_occupied:
            yield port
        port -= 1
        if port < 1024:
            port = 65535


def process_config_file(config_file, thread_id):
    """Modify config file.

    Args:
        config_file (str): Path of the original config file.
        thread_id (int): The ID of thread
    """

    with open(config_file, 'r', encoding='utf-8') as f:
        data = f.read()

    data = data.replace('# data_root', 'data_root')
    data = data.replace('# save_dir', 'save_dir')
    data = data.replace('# bg_dir', 'bg_dir')

    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(data)  # Will be automatically restored
    basename = osp.basename(config_file)
    save_config = osp.join(LOG_DIR, f'{thread_id:03d}_{basename}')
    with open(save_config, 'w', encoding='utf-8') as f:
        f.write(data)


def slurm_test(info: dict, thread_id, allotted_port):
    """Slurm test.

    Args:
        info (dict): Test info from metafile.yml
        thread_id (int): The ID of thread
        allotted_port (int): The ID of allotted port
    """

    sem.acquire()

    config = info['Config']
    weights = info['Weights']

    process_config_file(config, thread_id)
    basename, _ = osp.splitext(osp.basename(config))

    weights = osp.join(DOWNLOAD_DIR, 'hub', 'checkpoints',
                       osp.basename(weights))

    env_cmd = f'TORCH_HOME={DOWNLOAD_DIR} MASTER_PORT={allotted_port} '
    env_cmd += 'GPUS=2 GPUS_PER_NODE=2 CPUS_PER_TASK=8'
    base_cmd = 'bash tools/slurm_test.sh'
    task_cmd = f'{PARTITION} {basename}'
    out_file = osp.join(LOG_DIR, f'{thread_id:03d}_{basename}.log')
    cmd = f'{env_cmd} {base_cmd} {task_cmd} {config} {weights} &> {out_file}'

    print(f'RUN {thread_id:03d}: {cmd}')
    os.system(cmd)

    sem.release()


def test_models(meta_file, available_ports):
    """Download all pth files.

    Args:
        pth_files (List[str]): List of pth files.
    """

    global thread_num

    with open(meta_file, 'r', encoding='utf-8') as f:
        data = f.read()
    yaml_data = yaml.load(data, yaml.FullLoader)
    infos: list = yaml_data['Models']
    infos.sort(key=lambda info: info['Config'])

    for info in infos:
        if filter(info=info):
            allotted_port = next(available_ports)
            threading.Thread(
                target=slurm_test,
                args=(info, thread_num, allotted_port)).start()
            thread_num += 1


if __name__ == '__main__':

    assert 'nothing to commit, working tree clean' in os.popen(
        'git status').read(), 'Git: Please commit all changes first.'

    if len(sys.argv) <= 1:
        configs_root = osp.join(MMEditing_ROOT, 'configs')
        file_list = glob.glob(
            osp.join(configs_root, '**', '*metafile.yml'), recursive=True)
        file_list.sort()
    else:
        file_list = [
            fn for fn in sys.argv[1:] if osp.basename(fn) == 'metafile.yml'
        ]

    if not file_list:
        sys.exit(0)

    if not osp.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    thread_num = 0
    available_ports = find_available_port()
    for fn in file_list:
        test_models(fn, available_ports)

    os.system('git checkout .')
