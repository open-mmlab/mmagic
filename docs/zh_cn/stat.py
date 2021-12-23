#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
import functools as func
import glob
import re
from os.path import basename, splitext

import numpy as np
import titlecase


def anchor(name):
    return re.sub(r'-+', '-',
                  re.sub(r'[^a-zA-Z0-9\+]', '-',
                         name.strip().lower())).strip('-')


# Count algorithms

files = sorted(glob.glob('*_models.md'))
# files = sorted(glob.glob('docs/*_models.md'))

stats = []

for f in files:
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
        print(p)
        paperlinks[p] = ' '.join(
            (f'[⇨]({splitext(basename(f))[0]}.html#{anchor(paperlink)})'
             for paperlink in re.findall(
                 rf'\btitle\s*=\s*{{\s*{p}\s*}}.*?\n## (.*?)\s*[,;]?\s*\n',
                 revcontent, re.DOTALL | re.IGNORECASE)))
        print('   ', paperlinks[p])
    paperlist = '\n'.join(
        sorted(f'    - [{t}] {x} ({paperlinks[x]})' for t, x in papers))
    # count configs
    configs = set(x.lower().strip()
                  for x in re.findall(r'https.*configs/.*\.py', content))

    # count ckpts
    ckpts = set(x.lower().strip()
                for x in re.findall(r'https://download.*\.pth', content)
                if 'mmedit' in x)

    statsmsg = f"""
## [{title}]({f})

* 模型权重文件数量: {len(ckpts)}
* 配置文件数量: {len(configs)}
* 论文数量: {len(papers)}
{paperlist}

    """

    stats.append((papers, configs, ckpts, statsmsg))

allpapers = func.reduce(lambda a, b: a.union(b), [p for p, _, _, _ in stats])
allconfigs = func.reduce(lambda a, b: a.union(b), [c for _, c, _, _ in stats])
allckpts = func.reduce(lambda a, b: a.union(b), [c for _, _, c, _ in stats])

# Summarize

msglist = '\n'.join(x for _, _, _, x in stats)
papertypes, papercounts = np.unique([t for t, _ in allpapers],
                                    return_counts=True)
countstr = '\n'.join(
    [f'   - {t}: {c}' for t, c in zip(papertypes, papercounts)])

modelzoo = f"""
# 总览

* 模型权重文件数量: {len(allckpts)}
* 配置文件数量: {len(allconfigs)}
* 论文数量: {len(allpapers)}
{countstr}

有关支持的数据集，请参阅 [数据集总览](datasets.md)。

{msglist}

"""

with open('modelzoo.md', 'w') as f:
    f.write(modelzoo)

# Count datasets

files = sorted(glob.glob('*_datasets.md'))

datastats = []

for f in files:
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
        print(p)
        paperlinks[p] = ', '.join(
            (f'[{p} ⇨]({splitext(basename(f))[0]}.html#{anchor(p)})'
             for p in re.findall(
                 rf'\btitle\s*=\s*{{\s*{p}\s*}}.*?\n## (.*?)\s*[,;]?\s*\n',
                 revcontent, re.DOTALL | re.IGNORECASE)))
        print('   ', paperlinks[p])
    paperlist = '\n'.join(
        sorted(f'    - [{t}] {x} ({paperlinks[x]})' for t, x in papers))
    # count configs
    configs = set(x.lower().strip()
                  for x in re.findall(r'https.*configs/.*\.py', content))

    # count ckpts
    ckpts = set(x.lower().strip()
                for x in re.findall(r'https://download.*\.pth', content)
                if 'mmedit' in x)

    statsmsg = f"""
## [{title}]({f})

* 论文数量: {len(papers)}
{paperlist}

    """

    datastats.append((papers, configs, ckpts, statsmsg))

alldatapapers = func.reduce(lambda a, b: a.union(b),
                            [p for p, _, _, _ in datastats])

# Summarize

msglist = '\n'.join(x for _, _, _, x in stats)
datamsglist = '\n'.join(x for _, _, _, x in datastats)
papertypes, papercounts = np.unique([t for t, _ in alldatapapers],
                                    return_counts=True)
countstr = '\n'.join(
    [f'   - {t}: {c}' for t, c in zip(papertypes, papercounts)])

modelzoo = f"""
# 总览

* 论文数量: {len(alldatapapers)}
{countstr}

有关支持的算法, 可参见 [模型总览](modelzoo.md).

{datamsglist}
"""

with open('datasets.md', 'w') as f:
    f.write(modelzoo)
