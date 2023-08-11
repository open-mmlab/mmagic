#!/usr/bin/env python
import os
from glob import glob
from os import path as osp
from pathlib import Path

from modelindex.load_model_index import load
from tqdm import tqdm

MMAGIC_ROOT = Path(__file__).absolute().parents[3]
TARGET_ROOT = Path(__file__).absolute().parents[1] / 'model_zoo'


def write_file(file, content):
    os.makedirs(osp.dirname(file), exist_ok=True)
    with open(file, 'w', encoding='utf-8') as f:
        f.write(content)


def update_model_zoo():
    """load collections and models from model index, return summary,
    collections and models."""
    model_index_file = MMAGIC_ROOT / 'model-index.yml'
    model_index = load(str(model_index_file))
    model_index.build_models_with_collections()

    # parse model_index according to task
    tasks = {}
    full_models = set()
    for model in model_index.models:
        full_models.add(model.full_model)
        for r in model.results:
            _task = r.task.lower().split(', ')
            for t in _task:
                if t not in tasks:
                    tasks[t] = set()
                tasks[t].add(model.full_model)

    # assert the number of configs with the number of files
    collections = set([m.in_collection for m in full_models])
    assert len(collections) == len(os.listdir(MMAGIC_ROOT / 'configs')) - 1

    configs = set([str(MMAGIC_ROOT / m.config) for m in full_models])
    base_configs = glob(
        str(MMAGIC_ROOT / 'configs/_base_/**/*.py'), recursive=True)
    all_configs = glob(str(MMAGIC_ROOT / 'configs/**/*.py'), recursive=True)
    valid_configs = set(all_configs) - set(base_configs)
    untrackable_configs = valid_configs - configs
    assert len(untrackable_configs) == 0, '/n'.join(
        list(untrackable_configs)) + ' are not trackable.'

    # write for overview.md
    papers = set()
    checkpoints = set()
    for m in full_models:
        papers.add(m.paper['Title'])
        if m.weights is not None and m.weights.startswith('https:'):
            checkpoints.add(m.weights)
    task_desc = '\n'.join([
        f"  - [{t}]({t.replace('-', '_').replace(' ', '_')}.md)"
        for t in list(tasks.keys())
    ])

    # write overview.md
    overview = (f'# Overview\n\n'
                f'* Number of checkpoints: {len(checkpoints)}\n'
                f'* Number of configs: {len(configs)}\n'
                f'* Number of papers: {len(papers)}\n'
                f'  - ALGORITHM: {len(collections)}\n\n'
                f'* Tasks:\n{task_desc}')
    write_file(TARGET_ROOT / 'overview.md', overview)

    # write for index.rst
    task_desc = '\n'.join([
        f"    {t.replace('-', '_').replace(' ', '_')}.md"
        for t in list(tasks.keys())
    ])
    overview = (f'.. toctree::\n'
                f'    :maxdepth: 1\n'
                f'    :caption: Model Zoo\n\n'
                f'    overview.md\n'
                f'{task_desc}')
    write_file(TARGET_ROOT / 'index.rst', overview)

    # write for all the tasks
    for task, models in tqdm(tasks.items(), desc='create markdown files'):
        target_md = f"{task.replace('-', '_').replace(' ', '_')}.md"
        target_md = TARGET_ROOT / target_md
        models = sorted(models, key=lambda x: -x.data['Year'])

        checkpoints = set()
        for m in models:
            if m.weights is not None and m.weights.startswith('https:'):
                checkpoints.add(m.weights)
        collections = set([m.in_collection for m in models])

        papers = set()
        for m in models:
            papers.add(m.paper['Title'])

        content = ''
        readme = set()
        for m in models:
            if m.readme not in readme:
                readme.add(m.readme)
                with open(MMAGIC_ROOT / m.readme, 'r', encoding='utf-8') as f:
                    c = f.read()
                content += c.replace('# ', '## ')
        overview = (f'# {task}\n\n'
                    f'## Summary\n'
                    f'* Number of checkpoints: {len(checkpoints)}\n'
                    f'* Number of configs: {len(models)}\n'
                    f'* Number of papers: {len(papers)}\n'
                    f'  - ALGORITHM: {len(collections)}\n\n'
                    f'{content}')

        write_file(target_md, overview)


if __name__ == '__main__':
    update_model_zoo()
