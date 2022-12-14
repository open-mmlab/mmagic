import os

from tqdm import tqdm


def update_dataset_zoo():

    target_dir = 'dataset_zoo'
    source_dir = '../../tools/dataset_converters'
    os.makedirs(target_dir, exist_ok=True)

    # generate overview
    overviewmsg = """
# 概览

"""

    # generate index.rst
    rstmsg = """
.. toctree::
   :maxdepth: 1
   :caption: Dataset Zoo

   overview.md
"""

    subfolders = os.listdir(source_dir)
    for subf in tqdm(subfolders, desc='update dataset zoo'):

        target_subf = subf.replace('-', '_').lower()
        target_readme = os.path.join(target_dir, target_subf + '.md')
        source_readme = os.path.join(source_dir, subf, 'README_zh-CN.md')
        if not os.path.exists(source_readme):
            continue

        overviewmsg += f'\n- [{subf}]({target_subf}.md)'
        rstmsg += f'\n   {target_subf}.md'

        # generate all tasks dataset_zoo
        command = f'cat {source_readme} > {target_readme}'
        os.popen(command)

    with open(os.path.join(target_dir, 'overview.md'), 'w') as f:
        f.write(overviewmsg)

    with open(os.path.join(target_dir, 'index.rst'), 'w') as f:
        f.write(rstmsg)


if __name__ == '__main__':
    update_dataset_zoo()
