# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import platform
import unittest

import pytest
import torch
from mmengine import Config
from mmengine.runner import load_checkpoint

from mmedit.apis import (calculate_grid_size, colorization_inference,
                         init_model, inpainting_inference, matting_inference,
                         restoration_face_inference, restoration_inference,
                         restoration_video_inference, sample_conditional_model,
                         sample_img2img_model, sample_unconditional_model,
                         set_random_seed, video_interpolation_inference)
from mmedit.registry import MODELS
from mmedit.utils import register_all_modules, tensor2img

register_all_modules()


def test_init_model():
    set_random_seed(1)

    with pytest.raises(TypeError):
        init_model(['dog'])


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-cuda due to limited RAM.')
def test_colorization_inference():
    register_all_modules()

    if not torch.cuda.is_available():
        # RoI pooling only support in GPU
        return unittest.skip('test requires GPU and torch+cuda')

    if torch.cuda.is_available():
        device = torch.device('cuda', 0)
    else:
        device = torch.device('cpu')

    config = osp.join(
        osp.dirname(__file__),
        '../../..',
        'configs/inst_colorization/inst-colorizatioon_full_official_cocostuff-256x256.py'  # noqa
    )
    checkpoint = None

    cfg = Config.fromfile(config)
    model = MODELS.build(cfg.model)

    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)

    model.cfg = cfg
    model.to(device)
    model.eval()

    img_path = osp.join(
        osp.dirname(__file__), '..', '..',
        'data/image/img_root/horse/horse.jpeg')

    result = colorization_inference(model, img_path)
    assert tensor2img(result)[..., ::-1].shape == (256, 256, 3)


def test_unconditional_inference():
    cfg = osp.join(
        osp.dirname(__file__), '..', '..', '..', 'configs', 'dcgan',
        'dcgan_Glr4e-4_Dlr1e-4_1xb128-5kiters_mnist-64x64.py')
    cfg = Config.fromfile(cfg)
    model = MODELS.build(cfg.model)
    model.eval()

    # test num_samples can be divided by num_batches
    results = sample_unconditional_model(
        model, num_samples=4, sample_model='orig')
    assert results.shape == (4, 1, 64, 64)

    # test num_samples can not be divided by num_batches
    results = sample_unconditional_model(
        model, num_samples=4, num_batches=3, sample_model='orig')
    assert results.shape == (4, 1, 64, 64)


def test_conditional_inference():
    cfg = osp.join(
        osp.dirname(__file__), '..', '..', '..', 'configs', 'sngan_proj',
        'sngan-proj_woReLUinplace_lr2e-4-ndisc5-1xb64_cifar10-32x32.py')
    cfg = Config.fromfile(cfg)
    model = MODELS.build(cfg.model)
    model.eval()

    # test label is int
    results = sample_conditional_model(
        model, label=1, num_samples=4, sample_model='orig')
    assert results.shape == (4, 3, 32, 32)
    # test label is tensor
    results = sample_conditional_model(
        model,
        label=torch.FloatTensor([1.]),
        num_samples=4,
        sample_model='orig')
    assert results.shape == (4, 3, 32, 32)

    # test label is multi tensor
    results = sample_conditional_model(
        model,
        label=torch.FloatTensor([1., 2., 3., 4.]),
        num_samples=4,
        sample_model='orig')
    assert results.shape == (4, 3, 32, 32)

    # test label is list of int
    results = sample_conditional_model(
        model, label=[1, 2, 3, 4], num_samples=4, sample_model='orig')
    assert results.shape == (4, 3, 32, 32)

    # test label is None
    results = sample_conditional_model(
        model, num_samples=4, sample_model='orig')
    assert results.shape == (4, 3, 32, 32)

    # test label is invalid
    with pytest.raises(TypeError):
        results = sample_conditional_model(
            model, label='1', num_samples=4, sample_model='orig')

    # test length of label is not same as num_samples
    with pytest.raises(ValueError):
        results = sample_conditional_model(
            model, label=[1, 2], num_samples=4, sample_model='orig')

    # test num_samples can not be divided by num_batches
    results = sample_conditional_model(
        model, num_samples=3, num_batches=2, sample_model='orig')
    assert results.shape == (3, 3, 32, 32)


def test_inference():
    if torch.cuda.is_available():
        device = torch.device('cuda', 0)
    else:
        device = torch.device('cpu')

    data_root = osp.join(osp.dirname(__file__), '../../../')
    config = data_root + 'configs/dim/dim_stage3-v16-pln_1xb1-1000k_comp1k.py'
    checkpoint = 'https://download.openmmlab.com/mmediting/mattors/dim/dim_' +\
        'stage3_v16_pln_1x1_1000k_comp1k_SAD-50.6_20200609_111851-647f24b6.pth'

    img_path = data_root + 'tests/data/matting_dataset/merged/GT05.jpg'
    trimap_path = data_root + 'tests/data/matting_dataset/trimap/GT05.png'

    model = init_model(config, checkpoint, device=device)

    pred_alpha = matting_inference(model, img_path, trimap_path)
    assert pred_alpha.shape == (552, 800)


def test_inpainting_inference():
    register_all_modules()

    if torch.cuda.is_available():
        device = torch.device('cuda', 0)
    else:
        device = torch.device('cpu')

    checkpoint = None

    data_root = osp.join(osp.dirname(__file__), '../../')
    config_file = osp.join(data_root, 'configs', 'gl_test.py')

    cfg = Config.fromfile(config_file)
    model = MODELS.build(cfg.model_inference)

    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)

    model.cfg = cfg
    model.to(device)
    model.eval()

    masked_img_path = data_root + 'data/inpainting/celeba_test.png'
    mask_path = data_root + 'data/inpainting/bbox_mask.png'

    result = inpainting_inference(model, masked_img_path, mask_path)
    assert result.detach().cpu().numpy().shape == (3, 256, 256)


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-cuda due to limited RAM.')
def test_restoration_face_inference():
    if torch.cuda.is_available():
        device = torch.device('cuda', 0)
    else:
        device = torch.device('cpu')

    data_root = osp.join(osp.dirname(__file__), '../../../')
    config = data_root + 'configs/glean/glean_in128out1024_4xb2-300k_ffhq-celeba-hq.py'  # noqa

    checkpoint = None

    img_path = data_root + 'tests/data/image/face/000001.png'

    model = init_model(config, checkpoint, device=device)

    output = restoration_face_inference(model, img_path, 1, 1024)
    assert output.shape == (256, 256, 3)


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-cuda due to limited RAM.')
def test_restoration_inference():
    if torch.cuda.is_available():
        device = torch.device('cuda', 0)
    else:
        device = torch.device('cpu')

    data_root = osp.join(osp.dirname(__file__), '../../../')
    config = data_root + 'configs/esrgan/esrgan_x4c64b23g32_1xb16-400k_div2k.py'  # noqa
    checkpoint = None

    img_path = data_root + 'tests/data/image/lq/baboon_x4.png'

    model = init_model(config, checkpoint, device=device)

    output = restoration_inference(model, img_path)
    assert output.detach().cpu().numpy().shape == (3, 480, 500)


def test_restoration_video_inference():
    if torch.cuda.is_available():
        device = torch.device('cuda', 0)
    else:
        device = torch.device('cpu')

    data_root = osp.join(osp.dirname(__file__), '../../../')
    config = osp.join(data_root, 'configs/basicvsr/basicvsr_2xb4_reds4.py')
    checkpoint = None

    input_dir = osp.join(data_root, 'tests/data/frames/sequence/gt/sequence_1')

    model = init_model(config, checkpoint, device=device)

    output = restoration_video_inference(model, input_dir, 0, 0, '{:08d}.png',
                                         None)
    assert output.detach().numpy().shape == (1, 2, 3, 256, 448)

    input_video = data_root + 'tests/data/frames/test_inference.mp4'
    output = restoration_video_inference(
        model, input_video, 0, 0, '{:08d}.png', max_seq_len=3)

    output = restoration_video_inference(model, input_video, 3, 0,
                                         '{:08d}.png')


def test_translation_inference():
    cfg = osp.join(
        osp.dirname(__file__), '..', '..', '..', 'configs', 'pix2pix',
        'pix2pix_vanilla-unet-bn_1xb1-80kiters_facades.py')
    cfg = Config.fromfile(cfg)
    model = init_model(cfg, device='cpu')
    model.eval()
    data_path = osp.join(
        osp.dirname(__file__), '..', '..', 'data', 'unpaired', 'trainA',
        '1.jpg')
    # test num_samples can be divided by num_batches
    results = sample_img2img_model(
        model, image_path=data_path, target_domain='photo')
    print(results.shape)
    assert results.shape == (1, 3, 256, 256)

    # test target domain is None
    results = sample_img2img_model(model, image_path=data_path)
    assert results.shape == (1, 3, 256, 256)


def test_video_interpolation_inference():
    if torch.cuda.is_available():
        device = torch.device('cuda', 0)
    else:
        device = torch.device('cpu')

    data_root = osp.join(osp.dirname(__file__), '../../../')
    config = data_root + 'configs/cain/cain_g1b32_1xb5_vimeo90k-triplet.py'
    checkpoint = None

    model = init_model(config, checkpoint, device=device)

    input_dir = data_root + 'tests/data/frames/test_inference.mp4'
    video_interpolation_inference(
        model=model,
        input_dir=input_dir,
        output_dir='out/result_video.mp4',
        fps=60.0)

    input_dir = osp.join(data_root, 'tests/data/frames/sequence/gt/sequence_1')
    video_interpolation_inference(
        model=model, input_dir=input_dir, output_dir='out', fps=60.0)


def test_calculate_grid_size():
    inp_batch_size = (10, 13, 20, 1, 4)
    target_nrow = (4, 4, 5, 1, 2)
    for bz, tar in zip(inp_batch_size, target_nrow):
        assert calculate_grid_size(bz) == tar

    # test aspect_ratio is not None
    inp_batch_size = (10, 13, 20, 1, 4)
    aspect_ratio = (2, 3, 3, 4, 3)
    target_nrow = (3, 3, 3, 1, 2)
    for bz, ratio, tar in zip(inp_batch_size, aspect_ratio, target_nrow):
        assert calculate_grid_size(bz, ratio) == tar
