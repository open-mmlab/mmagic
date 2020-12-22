#!/usr/bin/env python
import functools as func
import glob
import os.path as osp
import re

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
        _papers = set(x.lower().strip()
                      for x in re.findall(r'\btitle\s*=\s*{(.*)}', content))

        ckpts = ckpts.union(_ckpts)
        papers = papers.union(_papers)

    paperlist = '\n'.join(sorted('    - ' + x for x in papers))

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

modelzoo_statsmsg = f"""
# Model Zoo

## Model Zoo Statistics
* Number of checkpoints: {len(allckpts)}
* Number of papers: {len(allpapers)}

{msglist}
"""

with open('model_zoo.md', 'r') as modelzoo_file:
    modelzoo_content = modelzoo_file.read()

modelzoo = modelzoo_statsmsg + modelzoo_content.replace('# Model Zoo', '', 1)

with open('modelzoo_statistics.md', 'w') as f:
    f.write(modelzoo)
