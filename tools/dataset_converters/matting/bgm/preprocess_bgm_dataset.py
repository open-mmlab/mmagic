# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from itertools import cycle

import mmcv


def generate_json(data_root, seg_root, bg_root, all_data):
    """Generate training json list for Background Matting video dataset.

    Args:
        data_root (str): Background Matting video data root.
        seg_root (str): Segmentation of Background Matting video data root.
        bg_root (str): Background video data root.
        all_data (bool): Whether use the last 80 frames of each video. If True,
            the last 80 frames will be added to training json. In the original
            Background Matting github repo, due to the use of motion cue, the
            last 80 frames is not used.
    """
    # use fixed-camera data to train the Background Matting model
    video_root = osp.join(data_root, 'fixed-camera/train')
    if seg_root is None:
        seg_root = video_root
    if bg_root is None:
        bg_root = osp.join(data_root, 'background')

    video_dirs = [
        entry for entry in os.listdir(video_root)
        if osp.isdir(osp.join(video_root, entry))
    ]
    bg_dirs = [
        entry for entry in os.listdir(bg_root)
        if osp.isdir(osp.join(bg_root, entry))
    ]

    # create an iterator that loops over all the background video frames
    bg_frames = []
    for bg_dir in bg_dirs:
        bg_frames.extend([
            osp.join(bg_root, bg_dir, f)
            for f in sorted(mmcv.scandir(osp.join(bg_root, bg_dir)))
        ])
    bg_stream = cycle(bg_frames)

    data_infos = []

    for video_dir in video_dirs:
        video_full_path = osp.join(video_root, video_dir)
        seg_full_path = osp.join(seg_root, video_dir)
        num_frames = len(
            list(mmcv.scandir(video_full_path, suffix='_img.png')))
        # In the original Background Matting github repo, the
        # last 80 frames is not used.
        effective_frames = num_frames if all_data else num_frames - 80
        for i in range(1, effective_frames + 1):
            # Though it's not composited, to be consistent with adobe data,
            # we call the captured image `merged`.
            # Since background video may not be under the same directory as
            # the Background Matting video data, we use full path in Background
            # Matting dataset annotation file.
            merged = osp.join(video_full_path, f'{i:04d}_img.png')
            seg = osp.join(seg_full_path, f'{i:04d}_masksDL.png')
            bg = video_full_path + '.png'
            bg_sup = next(bg_stream)
            data_info = dict(
                merged_path=merged,
                seg_path=seg,
                bg_path=bg,
                bg_sup_path=bg_sup)
            data_infos.append(data_info)
    save_json_path = 'fixed_camera_train.json'
    mmcv.dump(data_infos, osp.join(data_root, save_json_path))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare Background Matting video dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data_root', help='Background Matting video data root')
    parser.add_argument(
        '--seg-root',
        help='Segmentation of Background Matting video data root. If not '
        'specified, it will be considered segmentation results are placed in '
        'the video frames folder just as the original repo.')
    parser.add_argument(
        '--bg-root',
        help='Background video data root. If not specified, it will use the '
        'three background videos in the Captured_Data folder.')
    parser.add_argument(
        '--all-data',
        action='store_true',
        help='Also use the last 80 frames of each video')
    return parser.parse_args()


def main():
    args = parse_args()
    if not osp.exists(args.data_root):
        raise FileNotFoundError(f'{args.data_root} does not exist!')

    print('generating Background Matting dataset annotation file...')
    generate_json(args.data_root, args.seg_root, args.bg_root, args.all_data)
    print('annotation file generated...')


if __name__ == '__main__':
    main()
