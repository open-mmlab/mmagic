import os
import warnings
from pathlib import Path
from argparse import ArgumentParser

from mmedit.edit import MMEdit

def parse_args():
    parser = ArgumentParser()
    # input for matting
    parser.add_argument(
        '--img', 
        type=str, 
        default='', 
        help='Input image file or folder path.')

    # input for conditional models
    parser.add_argument(
        '--label', 
        type=int, 
        default=1, 
        help='Input label.')
    
    parser.add_argument(
        '--img-out-dir',
        type=str,
        default='resources/demo_results/unconditional/unconditional_samples_apis.png',
        help='Output directory of images.')
    parser.add_argument(
        '--model-name',
        type=str,
        default='styleganv1',
        help='Pretrained editing algorithm')
    parser.add_argument(
        '--model-version',
        type=str,
        default='a',
        help='Pretrained editing algorithm')
    parser.add_argument(
        '--model-config',
        type=str,
        default=None,
        help='Path to the custom config file of the selected editing model.')
    parser.add_argument(
        '--model-ckpt',
        type=str,
        default=None,
        help='Path to the custom checkpoint file of the selected det model.')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device used for inference.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the image in a popup window.')
    parser.add_argument(
        '--print-result',
        action='store_true',
        help='Whether to print the results.')
    parser.add_argument(
        '--pred-out-file',
        type=str,
        default='',
        help='File to save the inference results.')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    editor = MMEdit(**vars(args))
    editor.infer(**vars(args))

if __name__ == '__main__':
    main()