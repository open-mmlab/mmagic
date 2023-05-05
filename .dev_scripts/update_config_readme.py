#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.

# This tool is used to update README.md and README_zh-CN.md in configs

import glob
import os
import posixpath as osp  # Even on windows, use posixpath
import re
import sys

MMagic_ROOT = osp.dirname(osp.dirname(osp.dirname(__file__)))


def update_md(md_file):
    """Update README.md and README_zh-CN.md.

    Args:
        md_file (str): Path to .md file.
    Returns:
        Bool: If the target README.md file is different from the original.
    """
    # See https://github.com/open-mmlab/mmagic/pull/798 for these comments
    # unique_dict = generate_unique_name(md_file)

    md_file = md_file.replace(os.sep, '/')
    config_dir, _ = osp.split(md_file)
    files = os.listdir(config_dir)
    config_files = [file for file in files if file.endswith('.py')]

    for readme in ['README.md', 'README_zh-CN.md']:

        readme = osp.join(config_dir, readme)
        changed = False

        with open(readme, 'r', encoding='utf-8') as f:
            data = f.read()

        for config in config_files:

            _, ext = osp.splitext(config)
            if ext != '.py':
                continue

            re_config = config.replace('.', '\\.')
            re_config = config.replace('-', '\\-')
            re_result = re.search(rf'\]\(/(.*?)/{re_config}', data)
            if re_result is None:
                print(f'Warning: No {config} in {readme}')
                continue
            old_dir = re_result.groups()[0]
            if old_dir != config_dir:
                data = data.replace(old_dir, config_dir)
                print(f'from {old_dir} to {config_dir}')
                changed = True

        if changed:
            with open(readme, 'w', encoding='utf-8') as f:
                f.write(data)

    return False


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        configs_root = osp.join(MMagic_ROOT, 'configs')
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
    for fn in file_list:
        print(f'process {fn}')
        file_modified |= update_md(fn)

    sys.exit(1 if file_modified else 0)
