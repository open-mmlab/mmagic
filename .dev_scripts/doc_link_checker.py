# Copyright (c) MegFlow. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.
# /bin/python3

import argparse
import os
import re

import requests
from tqdm import tqdm


def make_parser():
    parser = argparse.ArgumentParser('Doc link checker')
    parser.add_argument(
        '--target',
        default='./docs',
        type=str,
        help='the directory or file to check')
    parser.add_argument(
        '--ignore', type=str, nargs='+', default=[], help='input image size')
    return parser


pattern = re.compile(r'\[.*?\]\(.*?\)')


def analyze_doc(home, path):
    problem_list = []
    code_block = 0
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith('```'):
                code_block = 1 - code_block

            if code_block > 0:
                continue

            if '[' in line and ']' in line and '(' in line and ')' in line:
                all = pattern.findall(line)
                for item in all:
                    # skip  ![]()
                    if item.find('[') == item.find(']') - 1:
                        continue

                    # process the case [text()]()
                    offset = item.find('](')
                    if offset == -1:
                        continue
                    item = item[offset:]
                    start = item.find('(')
                    end = item.find(')')
                    ref = item[start + 1:end]

                    if ref.startswith('http'):
                        if ref.startswith(
                                'https://download.openmmlab.com/'
                        ) or ref.startswith('http://download.openmmlab.com/'):
                            resp = requests.head(ref)
                            if resp.status_code == 200:
                                continue
                            else:
                                problem_list.append(ref)
                        else:
                            continue

                    if ref.startswith('#'):
                        continue

                    if ref == '<>':
                        continue

                    if '.md#' in ref:
                        ref = ref[:ref.find('#')]
                    if ref.startswith('/'):
                        fullpath = os.path.join(
                            os.path.dirname(__file__), '../', ref[1:])
                    else:
                        fullpath = os.path.join(home, ref)
                    if not os.path.exists(fullpath):
                        problem_list.append(ref)
            else:
                continue
    if len(problem_list) > 0:
        print(f'{path}:')
        for item in problem_list:
            print(f'\t {item}')
        print('\n')
        raise Exception('found link error')


def traverse(args):
    target = args.target
    if os.path.isfile(target):
        analyze_doc(os.path.dirname(target), target)
        return
    target_files = list(os.walk(target))
    target_files.sort()
    for home, dirs, files in tqdm(target_files):
        if home in args.ignore:
            continue
        for filename in files:
            if filename.endswith('.md'):
                path = os.path.join(home, filename)
                if os.path.islink(path) is False:
                    analyze_doc(home, path)


if __name__ == '__main__':
    args = make_parser().parse_args()
    traverse(args)
