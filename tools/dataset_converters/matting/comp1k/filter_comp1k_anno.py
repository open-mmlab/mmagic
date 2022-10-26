# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

import mmcv


def generate_json(comp1k_json_path, target_list_path, save_json_path):
    data_infos = mmcv.load(comp1k_json_path)
    targets = mmcv.list_from_file(target_list_path)
    new_data_infos = []
    for data_info in data_infos:
        for target in targets:
            if data_info['alpha_path'].endswith(target):
                new_data_infos.append(data_info)
                break

    mmcv.dump(new_data_infos, save_json_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Filter composition-1k annotation file')
    parser.add_argument(
        'comp1k_json_path',
        help='Path to the composition-1k dataset annotation file')
    parser.add_argument(
        'target_list_path',
        help='Path to the file name list that need to filter out')
    parser.add_argument(
        'save_json_path', help='Path to save the result json file')
    return parser.parse_args()


def main():
    args = parse_args()

    if not osp.exists(args.comp1k_json_path):
        raise FileNotFoundError(f'{args.comp1k_json_path} does not exist!')
    if not osp.exists(args.target_list_path):
        raise FileNotFoundError(f'{args.target_list_path} does not exist!')

    generate_json(args.comp1k_json_path, args.target_list_path,
                  args.save_json_path)

    print('Done!')


if __name__ == '__main__':
    main()
