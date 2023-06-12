import os
from argparse import ArgumentParser

import torch
from mmengine.registry import init_default_scope
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from vton_dataset import AlignedDataset

from mmagic.apis.inferencers.inference_functions import init_model
from projects.flow_style_vton.models import FlowStyleVTON

init_default_scope('mmagic')

config = 'configs/flow_style_vton_PFAFN_epoch_101.py'

parser = ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument(
    '--loadSize', type=int, default=512, help='scale images to this size')
parser.add_argument(
    '--fineSize', type=int, default=512, help='then crop to this size')
parser.add_argument('--dataroot', type=str, default='VITON_test')
parser.add_argument('--test_pairs', type=str, default='test_pairs.txt')
parser.add_argument(
    '--resize_or_crop',
    type=str,
    default='scale_width',
    help='scaling and cropping of images at load time \
    [resize_and_crop|crop|scale_width|scale_width_and_crop]')
parser.add_argument('--phase', type=str, default='test')
parser.add_argument('--isTrain', default=False)
parser.add_argument(
    '--no_flip',
    action='store_true',
    help='if specified, do not flip the images for data argumentation')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=4)

parser.add_argument('--output_dir', type=str, default='inference_results')
opt = parser.parse_args()

dataset = AlignedDataset(opt)
dataloader = DataLoader(dataset, opt.batch_size, num_workers=opt.num_workers)

device = torch.device(
    f'cuda:{opt.gpu_id}' if torch.cuda.is_available() else 'cpu')
# pretrained is set inside the config
model = init_model(config).to(device).eval()
assert isinstance(model, FlowStyleVTON)

os.makedirs('our_t_results', exist_ok=True)
os.makedirs('im_gar_flow_wg', exist_ok=True)
for i, data in enumerate(tqdm(dataloader)):
    p_tryon, combine = model.infer(data)
    save_image(
        p_tryon,
        os.path.join('our_t_results', data['p_name'][0]),
        nrow=int(1),
        normalize=True,
        value_range=(-1, 1))
    save_image(
        combine,
        os.path.join('im_gar_flow_wg', data['p_name'][0]),
        nrow=int(1),
        normalize=True,
        range=(-1, 1),
    )
