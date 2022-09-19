#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.

import functools as func
import glob
import os.path as osp
import re
from os.path import basename, dirname

import numpy as np
import titlecase
from tqdm import tqdm

github_link = 'https://github.com/open-mmlab/mmediting/blob/1.x/'


def anchor(name):
    return re.sub(r'-+', '-',
                  re.sub(r'[^a-zA-Z0-9\+]', '-',
                         name.strip().lower())).strip('-')


# Count algorithms
def update_model_zoo():

    root_dir = dirname(dirname(dirname(dirname(osp.abspath(__file__)))))
    files = sorted(glob.glob(osp.join(root_dir, 'configs/*/README.md')))
    stats = []

    for f in tqdm(files, desc='update model zoo'):
        with open(f, 'r') as content_file:
            content = content_file.read()

        # title
        title = content.split('\n')[0].replace('#', '')

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
                                  'README.md')
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
        ckpts = set(ckpts)

        # count tasks
        task_desc = list(
            set(x.lower().strip()
                for x in re.findall(r'\*\*Task\*\*: .*', content)))
        tasks = set()
        if len(task_desc) > 0:
            tasks = set(task_desc[0].split('**task**: ')[1].split(', '))

        statsmsg = f"""## {title}"""
        if len(tasks) > 0:
            statsmsg += f"\n* Tasks: {','.join(list(tasks))}"
        statsmsg += f"""
* Number of checkpoints: {len(ckpts)}
* Number of configs: {len(configs)}
* Number of papers: {len(papers)}
{paperlist}

"""

        # * We should have: {len(glob.glob(osp.join(dirname(f), '*.py')))}
        stats.append((papers, configs, ckpts, tasks, statsmsg))

    allpapers = func.reduce(lambda a, b: a.union(b),
                            [p for p, _, _, _, _ in stats])
    allconfigs = func.reduce(lambda a, b: a.union(b),
                             [c for _, c, _, _, _ in stats])
    allckpts = func.reduce(lambda a, b: a.union(b),
                           [c for _, _, c, _, _ in stats])
    alltasks = func.reduce(lambda a, b: a.union(b),
                           [t for _, _, _, t, _ in stats])
    task_desc = '\n    - '.join(list(alltasks))

    # Summarize

    msglist = '\n'.join(x for _, _, _, _, x in stats)
    papertypes, papercounts = np.unique([t for t, _ in allpapers],
                                        return_counts=True)
    countstr = '\n'.join(
        [f'   - {t}: {c}' for t, c in zip(papertypes, papercounts)])
    countstr = '\n'.join([f'   - ALGORITHM: {len(stats)}'])

    modelzoo = f"""# Overview

* Number of checkpoints: {len(allckpts)}
* Number of configs: {len(allconfigs)}
* Number of papers: {len(allpapers)}
{countstr}
* Tasks:
    - {task_desc}

For supported datasets, see [datasets overview](dataset_zoo/0_overview.md).

{msglist}

    """

    with open('3_model_zoo.md', 'w') as f:
        f.write(modelzoo)


if __name__ == '__main__':
    update_model_zoo()
