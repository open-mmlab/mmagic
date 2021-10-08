from argparse import ArgumentParser

import requests
from PIL import Image


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('model_name', help='The model name in the server')
    parser.add_argument(
        '--inference-addr',
        default='127.0.0.1:8080',
        help='Address and port of the inference server')
    parser.add_argument(
        '--img-path',
        type=str,
        default='demo.png',
        help='Path to save generated image.')
    parser.add_argument(
        '--img-size', type=int, default=128, help='Size of the output image.')
    args = parser.parse_args()
    return args


def save_results(content, img_path, img_size):
    img = Image.frombytes('RGB', (img_size, img_size), content)
    img.save(img_path)


def main(args):
    url = 'http://' + args.inference_addr + '/predictions/' + args.model_name

    # just post a meanless dict
    response = requests.post(url, {'key': 'value'})
    save_results(response.content, args.img_path, args.img_size)


if __name__ == '__main__':
    parsed_args = parse_args()
    main(parsed_args)
