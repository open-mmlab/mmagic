# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

import mmengine
import requests

RESOURCES = {
    'Matting': [
        'https://download.openmmlab.com/mmediting/resources/input/matting/GT05.jpg',  # noqa
        'https://download.openmmlab.com/mmediting/resources/input/matting/GT05_trimap.jpg',  # noqa
        'https://download.openmmlab.com/mmediting/resources/input/matting/readme.md'  # noqa
    ],
    'Inpainting': [
        'https://download.openmmlab.com/mmediting/resources/input/inpainting/bbox_mask.png',  # noqa
        'https://download.openmmlab.com/mmediting/resources/input/inpainting/celeba_test.png',  # noqa
        'https://download.openmmlab.com/mmediting/resources/input/inpainting/readme.md'  # noqa
    ],
    'Image Super-Resolution': [
        'https://download.openmmlab.com/mmediting/resources/input/restoration/000001.png',  # noqa
        'https://download.openmmlab.com/mmediting/resources/input/restoration/0901x2.png',  # noqa
        'https://download.openmmlab.com/mmediting/resources/input/restoration/readme.md'  # noqa
    ],
    'Image2Image Translation': [
        'https://download.openmmlab.com/mmediting/resources/input/translation/gt_mask_0.png',  # noqa
        'https://download.openmmlab.com/mmediting/resources/input/translation/readme.md'  # noqa
    ],
    'Video Interpolation': [
        'https://download.openmmlab.com/mmediting/resources/input/video_interpolation/b-3LLDhc4EU_000000_000010.mp4',  # noqa
        'https://download.openmmlab.com/mmediting/resources/input/video_interpolation/readme.md'  # noqa
    ],
    'Video Super-Resolution': [
        'https://download.openmmlab.com/mmediting/resources/input/video_restoration/QUuC4vJs_000084_000094_400x320.mp4',  # noqa
        'https://download.openmmlab.com/mmediting/resources/input/video_restoration/readme.md',  # noqa
        'https://download.openmmlab.com/mmediting/resources/input/video_restoration/v_Basketball_g01_c01.avi'  # noqa
    ]
}


def parse_args():
    parser = argparse.ArgumentParser(description='Download resources')
    parser.add_argument(
        '--root-dir',
        type=str,
        help='resource root dir',
        default='../resources')
    parser.add_argument(
        '--task',
        type=str,
        help='one specific task, if None : download all resources',
        default=None)
    parser.add_argument(
        '--print-all', action='store_true', help='print all resources')
    parser.add_argument(
        '--print-task-type', action='store_true', help='print all task types')
    parser.add_argument(
        '--print-task',
        type=str,
        help='print all tasks that need input resources',
        default=None)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.print_all:
        print('all inference resources:')
        for key in RESOURCES.keys():
            print(key)
            for value in RESOURCES[key]:
                print(value)
        return

    if args.print_task_type:
        print('all task type:')
        for key in RESOURCES.keys():
            print(key)
        return

    if args.print_task:
        print('RESOURCES of task ' + args.print_task + ':')
        for value in RESOURCES[args.print_task]:
            print(value)
        return

    to_be_download = []
    if args.task and args.task in RESOURCES.keys():
        to_be_download.extend(RESOURCES[args.task])
    else:
        for key in RESOURCES.keys():
            to_be_download.extend(RESOURCES[key])

    put_root_path = osp.join(osp.dirname(__file__), args.root_dir)
    for item in to_be_download:
        item_relative_path = item[item.find('input'):]
        put_path = osp.join(put_root_path, item_relative_path)
        mmengine.mkdir_or_exist(osp.dirname(put_path))
        response = requests.get(item)
        open(put_path, 'wb').write(response.content)
        print('Download finished: ' + item + ' to ' + put_path)


if __name__ == '__main__':
    main()
