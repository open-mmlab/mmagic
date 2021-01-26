#!/usr/bin/env python
import functools as func
import glob
import os.path as osp
import re

import numpy as np

config_dpaths = sorted(glob.glob('../configs/**'))

stats = []

for config_dpath in config_dpaths:
    files = sorted(glob.glob(osp.join(config_dpath, '*/README.md')))

    title = config_dpath.replace('../configs/', '', 1)
    ckpts = set()
    papers = set()

    for f in files:
        with open(f, 'r') as content_files:
            content = content_files.read()

        # count ckpts
        _ckpts = set(x.lower().strip()
                     for x in re.findall(r'https://download.*\.pth', content)
                     if 'mmediting' in x)
        _papers = set(
            (papertype, paper.lower().strip())
            for (papertype, paper) in re.findall(
                r'\[([A-Z]+?)\].*?\btitle\s*=\s*{(.*?)}', content, re.DOTALL))

        ckpts = ckpts.union(_ckpts)
        papers = papers.union(_papers)

    paperlist = '\n'.join(sorted(f'    - [{t}] {x}' for t, x in papers))

    statsmsg = f"""
### {title.title()}

* Number of checkpoints: {len(ckpts)}
* Number of papers: {len(papers)}
{paperlist}

 """

    stats.append((papers, ckpts, statsmsg))

allpapers = func.reduce(lambda a, b: a.union(b), [p for p, _, _ in stats])
allckpts = func.reduce(lambda a, b: a.union(b), [c for _, c, _ in stats])
msglist = '\n'.join(x for _, _, x in stats)

papertypes, papercounts = np.unique([t for t, _ in allpapers],
                                    return_counts=True)
countstr = '\n'.join(
    [f'   - {t}: {c}' for t, c in zip(papertypes, papercounts)])

modelzoo_statsmsg = f"""
# Model Zoo Statistics
* Number of checkpoints: {len(allckpts)}
* Number of papers: {len(allpapers)}
{countstr}
{msglist}
"""

with open('modelzoo_statistics.md', 'w') as f:
    f.write(modelzoo_statsmsg)
