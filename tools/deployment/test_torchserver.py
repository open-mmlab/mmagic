# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import cv2
import numpy as np
import requests
from PIL import Image


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('model_name', help='The model name in the server')
    parser.add_argument(
        '--inference-addr',
        default='127.0.0.1:8080',
        help='Address and port of the inference server')
    parser.add_argument('--img-path', type=str, help='The input LQ image.')
    parser.add_argument(
        '--save-path', type=str, help='Path to save the generated GT image.')
    args = parser.parse_args()
    return args


def save_results(content, save_path, ori_shape):
    ori_len = np.prod(ori_shape)
    scale = int(np.sqrt(len(content) / ori_len))
    target_size = [int(size * scale) for size in ori_shape[:2][::-1]]
    # Convert to RGB and save image
    img = Image.frombytes('RGB', target_size, content, 'raw', 'BGR', 0, 0)
    img.save(save_path)


def main(args):
    url = 'http://' + args.inference_addr + '/predictions/' + args.model_name
    ori_shape = cv2.imread(args.img_path).shape
    with open(args.img_path, 'rb') as image:
        response = requests.post(url, image)
    save_results(response.content, args.save_path, ori_shape)


if __name__ == '__main__':
    parsed_args = parse_args()
    main(parsed_args)
