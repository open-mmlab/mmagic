#!/usr/bin/env python
import functools as func
import glob
import os.path as osp
import re

url_prefix = 'https://github.com/open-mmlab/mmediting/blob/master/'

inpainting_files = sorted(glob.glob('../configs/inpainting/*/README.md'))
mattor_files = sorted(glob.glob('../configs/mattors/*/README.md'))
restorer_files = sorted(glob.glob('../configs/restores/*/README.md'))
synthesizer_files = sorted(glob.glob('../configs/synthesizer/*/README.md'))

all_files = (inpainting_files, mattor_files, restorer_files, synthesizer_files)
titles = ('Inpainting', 'Mattor', 'Restorer', 'synthesizer')

stats = []
papers = set()
total_num_ckpts = 0

for title, files in zip(titles, all_files):
    ckpts = set()
    msg_list = []
    for f in files:
        url = osp.dirname(f.replace('../', url_prefix))

        with open(f, 'r') as content_file:
            content = content_file.read()

        paper = set([
            content.split('\n')[0].replace('#', ''),
        ])
        papers.union(paper)
        ckpts_ = set(x.lower().strip()
                     for x in re.findall(r'https?://download.*\.pth', content)
                     if 'mmediting' in x)

        msg = f"""\t* [{paper}]({url}) ({len(ckpts_)} ckpts)"""

        ckpts = ckpts.union(ckpts_)
        msg_list.append(msg)
    msg = '\n'.join(msg_list)
    statsmsg = f"""
## {title}

* Number of checkpoints: {len(ckpts)}
* Number of papers: {len(papers)}
{msg}
"""
    stats.append((papers, ckpts, statsmsg))

allpapers = func.reduce(lambda a, b: a.union(b), [p for p, _, _ in stats])
allckpts = func.reduce(lambda a, b: a.union(b), [c for _, c, _ in stats])
msglist = '\n'.join(x for _, _, x in stats)

modelzoo = f"""
## Model Zoo Statistics

* Number of papers: {len(allpapers)}
* Number of checkpoints: {len(allckpts)}
{msglist}
"""

with open('model_zoo.md', 'a') as f:
    f.write(modelzoo)
