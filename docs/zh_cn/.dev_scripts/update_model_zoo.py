#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.

import functools as func
import glob
import os
import os.path as osp
import re
from os.path import basename, dirname

import numpy as np
import titlecase
from tqdm import tqdm

github_link = 'https://github.com/open-mmlab/mmagic/blob/main/'


def anchor(name):
    return re.sub(r'-+', '-',
                  re.sub(r'[^a-zA-Z0-9\+]', '-',
                         name.strip().lower())).strip('-')


def summarize(stats, name):
    allpapers = func.reduce(lambda a, b: a.union(b),
                            [p for p, _, _, _, _, _, _ in stats])
    allconfigs = func.reduce(lambda a, b: a.union(b),
                             [c for _, c, _, _, _, _, _ in stats])
    allckpts = func.reduce(lambda a, b: a.union(b),
                           [c for _, _, c, _, _, _, _ in stats])
    alltasks = func.reduce(lambda a, b: a.union(b),
                           [t for _, _, _, t, _, _, _ in stats])
    task_desc = '\n'.join([
        f"    - [{task}]({task.replace('-', '_').replace(' ', '_').lower()}.md)"  # noqa
        for task in list(alltasks)
    ])

    # Overview
    papertypes, papercounts = np.unique([t for t, _ in allpapers],
                                        return_counts=True)
    countstr = '\n'.join(
        [f'   - {t}: {c}' for t, c in zip(papertypes, papercounts)])
    countstr = '\n'.join([f'   - ALGORITHM: {len(stats)}'])

    summary = f"""# {name}
"""

    if name != 'Overview':
        summary += '\n## 概览'

    summary += f"""
* 预训练权重个数: {len(allckpts)}
* 配置文件个数: {len(allconfigs)}
* 论文个数: {len(allpapers)}
{countstr}
    """

    if name == 'Overview':
        summary += f"""
* 任务:
{task_desc}

"""

    return summary


# Count algorithms
def update_model_zoo():

    target_dir = 'model_zoo'

    os.makedirs(target_dir, exist_ok=True)

    root_dir = dirname(dirname(dirname(dirname(osp.abspath(__file__)))))
    files = sorted(glob.glob(osp.join(root_dir, 'configs/*/README_zh-CN.md')))
    stats = []

    for f in tqdm(files, desc='update model zoo'):
        with open(f, 'r') as content_file:
            content = content_file.read()

        # title
        title = content.split('\n')[0].replace('#', '')
        year = title.split('\'')[-1].split(')')[0]

        # count papers
        papers = set(
            (papertype,
             titlecase.titlecase(paper.lower().strip()).replace('+', r'\+'))
            for (papertype, paper) in re.findall(
                r'<!--\s*\[([A-Z]*?)\]\s*-->\s*\n.*?\btitle\s*=\s*{(.*?)}',
                content, re.DOTALL))

        # paper links
        revcontent = '\n'.join(list(reversed(content.splitlines())))
        paperlinks = {}
        for _, p in papers:
            paper_link = osp.join(github_link, 'configs', basename(dirname(f)),
                                  'README_zh-CN.md')
            # print(p, paper_link)
            paperlinks[p] = ' '.join(
                (f'[⇨]({paper_link}#{anchor(paperlink)})'
                 for paperlink in re.findall(
                     rf'\btitle\s*=\s*{{\s*{p}\s*}}.*?\n## (.*?)\s*[,;]?\s*\n',
                     revcontent, re.DOTALL | re.IGNORECASE)))
            # print('   ', paperlinks[p])
        paperlist = '\n'.join(
            sorted(f'    - [{t}] {x} ({paperlinks[x]})' for t, x in papers))

        # count configs
        configs = set(x.lower().strip()
                      for x in re.findall(r'/configs/.*?\.py', content))

        # count ckpts
        ckpts = list(
            x.lower().strip()
            for x in re.findall(r'\[model\]\(https\:\/\/.*\.pth', content))
        ckpts.extend(
            x.lower().strip()
            for x in re.findall(r'\[ckpt\]\(https\:\/\/.*\.pth', content))
        ckpts.extend(
            x.lower().strip()
            for x in re.findall(r'\[模型\]\(https\:\/\/.*\.pth', content))
        ckpts.extend(
            x.lower().strip()
            for x in re.findall(r'\[权重\]\(https\:\/\/.*\.pth', content))
        ckpts = set(ckpts)

        # count tasks
        task_desc = list(
            set(x.lower().strip()
                for x in re.findall(r'\*\*任务\*\*: .*', content)))
        tasks = set()
        if len(task_desc) > 0:
            tasks = set(task_desc[0].split('**任务**: ')[1].split(', '))

        statsmsg = f"""## {title}"""
        if len(tasks) > 0:
            statsmsg += f"\n* Tasks: {','.join(list(tasks))}"
        statsmsg += f"""

* 预训练权重个数: {len(ckpts)}
* 配置文件个数: {len(configs)}
* 论文个数: {len(papers)}
{paperlist}

"""
        # * We should have: {len(glob.glob(osp.join(dirname(f), '*.py')))}
        content = content.replace('# ', '## ')
        stats.append((papers, configs, ckpts, tasks, year, statsmsg, content))

    # overview
    overview = summarize(stats, '概览')
    with open(osp.join(target_dir, 'overview.md'), 'w') as f:
        f.write(overview)

    alltasks = func.reduce(lambda a, b: a.union(b),
                           [t for _, _, _, t, _, _, _ in stats])

    # index.rst
    indexmsg = """
.. toctree::
   :maxdepth: 1
   :caption: 模型库

   overview.md
"""

    for task in alltasks:
        task = task.replace(' ', '_').replace('-', '_').lower()
        indexmsg += f'   {task}.md\n'

    with open(osp.join(target_dir, 'index.rst'), 'w') as f:
        f.write(indexmsg)

    #  task-specific
    for task in alltasks:
        filtered_model = [
            (paper, config, ckpt, tasks, year, x, content)
            for paper, config, ckpt, tasks, year, x, content in stats
            if task in tasks
        ]
        filtered_model = sorted(filtered_model, key=lambda x: x[-3])[::-1]
        overview = summarize(filtered_model, task)

        msglist = '\n'.join(x for _, _, _, _, _, _, x in filtered_model)
        task = task.replace(' ', '_').replace('-', '_').lower()
        with open(osp.join(target_dir, f'{task}.md'), 'w') as f:
            f.write(overview + '\n' + msglist)


if __name__ == '__main__':
    update_model_zoo()
