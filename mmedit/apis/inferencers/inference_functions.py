# Copyright (c) OpenMMLab. All rights reserved.
import glob
import math
import os
import os.path as osp

import cv2
import mmcv
import numpy as np
import torch
from mmengine import Config, is_list_of
from mmengine.config import ConfigDict
from mmengine.dataset import Compose
from mmengine.dataset.utils import default_collate as collate
from mmengine.fileio import FileClient
from mmengine.runner import load_checkpoint
from mmengine.runner import set_random_seed as set_random_seed_engine
from mmengine.utils import ProgressBar
from torch.nn.parallel import scatter

from mmedit.models.base_models import BaseTranslationModel
from mmedit.registry import MODELS
from mmedit.utils import register_all_modules

VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi')
FILE_CLIENT = FileClient('disk')


def set_random_seed(seed, deterministic=False, use_rank_shift=True):
    """Set random seed.

    In this function, we just modify the default behavior of the similar
    function defined in MMCV.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
        rank_shift (bool): Whether to add rank number to the random seed to
            have different random seed in different threads. Default: True.
    """
    set_random_seed_engine(seed, deterministic, use_rank_shift)


def delete_cfg(cfg, key='init_cfg'):
    """Delete key from config object.

    Args:
        cfg (str or :obj:`mmengine.Config`): Config object.
        key (str): Which key to delete.
    """

    if key in cfg:
        cfg.pop(key)
    for _key in cfg.keys():
        if isinstance(cfg[_key], ConfigDict):
            delete_cfg(cfg[_key], key)


def init_model(config, checkpoint=None, device='cuda:0'):
    """Initialize a model from config file.

    Args:
        config (str or :obj:`mmengine.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str): Which device the model will deploy. Default: 'cuda:0'.

    Returns:
        nn.Module: The constructed model.
    """

    if isinstance(config, str):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    # config.test_cfg.metrics = None
    delete_cfg(config.model, 'init_cfg')

    register_all_modules()
    model = MODELS.build(config.model)

    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)

    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def sample_unconditional_model(model,
                               num_samples=16,
                               num_batches=4,
                               sample_model='ema',
                               **kwargs):
    """Sampling from unconditional models.

    Args:
        model (nn.Module): Unconditional models in MMGeneration.
        num_samples (int, optional): The total number of samples.
            Defaults to 16.
        num_batches (int, optional): The number of batch size for inference.
            Defaults to 4.
        sample_model (str, optional): Which model you want to use. ['ema',
            'orig']. Defaults to 'ema'.

    Returns:
        Tensor: Generated image tensor.
    """
    # set eval mode
    model.eval()
    # construct sampling list for batches
    n_repeat = num_samples // num_batches
    batches_list = [num_batches] * n_repeat

    if num_samples % num_batches > 0:
        batches_list.append(num_samples % num_batches)
    res_list = []

    # inference
    for batches in batches_list:
        res = model(
            dict(num_batches=batches, sample_model=sample_model), **kwargs)
        res_list.extend([item.fake_img.data.cpu() for item in res])

    results = torch.stack(res_list, dim=0)
    return results


@torch.no_grad()
def sample_conditional_model(model,
                             num_samples=16,
                             num_batches=4,
                             sample_model='ema',
                             label=None,
                             **kwargs):
    """Sampling from conditional models.

    Args:
        model (nn.Module): Conditional models in MMGeneration.
        num_samples (int, optional): The total number of samples.
            Defaults to 16.
        num_batches (int, optional): The number of batch size for inference.
            Defaults to 4.
        sample_model (str, optional): Which model you want to use. ['ema',
            'orig']. Defaults to 'ema'.
        label (int | torch.Tensor | list[int], optional): Labels used to
            generate images. Default to None.,

    Returns:
        Tensor: Generated image tensor.
    """
    # set eval mode
    model.eval()
    # construct sampling list for batches
    n_repeat = num_samples // num_batches
    batches_list = [num_batches] * n_repeat

    # check and convert the input labels
    if isinstance(label, int):
        label = torch.LongTensor([label] * num_samples)
    elif isinstance(label, torch.Tensor):
        label = label.type(torch.int64)
        if label.numel() == 1:
            # repeat single tensor
            # call view(-1) to avoid nested tensor like [[[1]]]
            label = label.view(-1).repeat(num_samples)
        else:
            # flatten multi tensors
            label = label.view(-1)
    elif isinstance(label, list):
        if is_list_of(label, int):
            label = torch.LongTensor(label)
            # `nargs='+'` parse single integer as list
            if label.numel() == 1:
                # repeat single tensor
                label = label.repeat(num_samples)
        else:
            raise TypeError('Only support `int` for label list elements, '
                            f'but receive {type(label[0])}')
    elif label is None:
        pass
    else:
        raise TypeError('Only support `int`, `torch.Tensor`, `list[int]` or '
                        f'None as label, but receive {type(label)}.')

    # check the length of the (converted) label
    if label is not None and label.size(0) != num_samples:
        raise ValueError('Number of elements in the label list should be ONE '
                         'or the length of `num_samples`. Requires '
                         f'{num_samples}, but receive {label.size(0)}.')

    # make label list
    label_list = []
    for n in range(n_repeat):
        if label is None:
            label_list.append(None)
        else:
            label_list.append(label[n * num_batches:(n + 1) * num_batches])

    if num_samples % num_batches > 0:
        batches_list.append(num_samples % num_batches)
        if label is None:
            label_list.append(None)
        else:
            label_list.append(label[(n + 1) * num_batches:])

    res_list = []

    # inference
    for batches, labels in zip(batches_list, label_list):
        res = model(
            dict(
                num_batches=batches, labels=labels, sample_model=sample_model),
            **kwargs)
        res_list.extend([item.fake_img.data.cpu() for item in res])
    results = torch.stack(res_list, dim=0)
    return results


def inpainting_inference(model, masked_img, mask):
    """Inference image with the model.

    Args:
        model (nn.Module): The loaded model.
        masked_img (str): File path of image with mask.
        mask (str): Mask file path.

    Returns:
        Tensor: The predicted inpainting result.
    """
    device = next(model.parameters()).device  # model device

    # build the data pipeline
    infer_pipeline = [
        dict(type='LoadImageFromFile', key='gt', channel_order='bgr'),
        dict(
            type='LoadMask',
            mask_mode='file',
        ),
        dict(type='GetMaskedImage'),
        dict(type='PackEditInputs'),
    ]

    test_pipeline = Compose(infer_pipeline)
    # prepare data
    data = dict(gt_path=masked_img, mask_path=mask)
    _data = test_pipeline(data)
    data = dict()
    data['inputs'] = _data['inputs'] / 255.0
    data = collate([data])
    data['data_samples'] = [_data['data_samples']]
    if 'cuda' in str(device):
        data = scatter(data, [device])[0]
        data['data_samples'][0].mask.data = scatter(
            data['data_samples'][0].mask.data, [device])[0] / 255.0
    # else:
    #     data.pop('meta')
    # forward the model
    with torch.no_grad():
        result, x = model(mode='tensor', **data)

    masks = _data['data_samples'].mask.data * 255
    masked_imgs = data['inputs'][0]
    result = result[0] * masks + masked_imgs * (1. - masks)
    return result


def matting_inference(model, img, trimap):
    """Inference image(s) with the model.

    Args:
        model (nn.Module): The loaded model.
        img (str): Image file path.
        trimap (str): Trimap file path.

    Returns:
        np.ndarray: The predicted alpha matte.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # remove alpha from test_pipeline
    keys_to_remove = ['alpha', 'ori_alpha']
    for key in keys_to_remove:
        for pipeline in list(cfg.test_pipeline):
            if 'key' in pipeline and key == pipeline['key']:
                cfg.test_pipeline.remove(pipeline)
            if 'keys' in pipeline and key in pipeline['keys']:
                pipeline['keys'].remove(key)
                if len(pipeline['keys']) == 0:
                    cfg.test_pipeline.remove(pipeline)
            if 'meta_keys' in pipeline and key in pipeline['meta_keys']:
                pipeline['meta_keys'].remove(key)
    # build the data pipeline
    test_pipeline = Compose(cfg.test_pipeline)
    # prepare data
    data = dict(merged_path=img, trimap_path=trimap)
    _data = test_pipeline(data)
    trimap = _data['data_samples'].trimap.data
    data = dict()
    data['inputs'] = torch.cat([_data['inputs'], trimap], dim=0).float()
    data = collate([data])
    data['data_samples'] = [_data['data_samples']]
    if 'cuda' in str(device):
        data = scatter(data, [device])[0]
    # forward the model
    with torch.no_grad():
        result = model(mode='predict', **data)
    result = result[0].output
    result = result.pred_alpha.data
    return result.cpu().numpy()


def sample_img2img_model(model, image_path, target_domain=None, **kwargs):
    """Sampling from translation models.

    Args:
        model (nn.Module): The loaded model.
        image_path (str): File path of input image.
        style (str): Target style of output image.
    Returns:
        Tensor: Translated image tensor.
    """
    assert isinstance(model, BaseTranslationModel)

    # get source domain and target domain
    if target_domain is None:
        target_domain = model._default_domain
    source_domain = model.get_other_domains(target_domain)[0]

    cfg = model.cfg
    # build the data pipeline
    test_pipeline = Compose(cfg.test_pipeline)

    # prepare data
    data = dict()
    # dirty code to deal with test data pipeline
    data['pair_path'] = image_path
    data[f'img_{source_domain}_path'] = image_path
    data[f'img_{target_domain}_path'] = image_path

    data = collate([test_pipeline(data)])
    data = model.data_preprocessor(data, False)
    inputs_dict = data['inputs']

    source_image = inputs_dict[f'img_{source_domain}']

    # forward the model
    with torch.no_grad():
        results = model(
            source_image,
            test_mode=True,
            target_domain=target_domain,
            **kwargs)
    output = results['target']
    return output


def restoration_inference(model, img, ref=None):
    """Inference image with the model.

    Args:
        model (nn.Module): The loaded model.
        img (str): File path of input image.
        ref (str | None): File path of reference image. Default: None.

    Returns:
        Tensor: The predicted restoration result.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    # select the data pipeline
    if cfg.get('demo_pipeline', None):
        test_pipeline = cfg.demo_pipeline
    elif cfg.get('test_pipeline', None):
        test_pipeline = cfg.test_pipeline
    else:
        test_pipeline = cfg.val_pipeline

    # remove gt from test_pipeline
    keys_to_remove = ['gt', 'gt_path']
    for key in keys_to_remove:
        for pipeline in list(test_pipeline):
            if 'key' in pipeline and key == pipeline['key']:
                test_pipeline.remove(pipeline)
            if 'keys' in pipeline and key in pipeline['keys']:
                pipeline['keys'].remove(key)
                if len(pipeline['keys']) == 0:
                    test_pipeline.remove(pipeline)
            if 'meta_keys' in pipeline and key in pipeline['meta_keys']:
                pipeline['meta_keys'].remove(key)
    # build the data pipeline
    test_pipeline = Compose(test_pipeline)
    # prepare data
    if ref:  # Ref-SR
        data = dict(img_path=img, ref_path=ref)
    else:  # SISR
        data = dict(img_path=img)
    _data = test_pipeline(data)
    data = dict()
    data['inputs'] = _data['inputs'] / 255.0
    data = collate([data])
    if ref:
        data['data_samples'] = [_data['data_samples']]
    if 'cuda' in str(device):
        data = scatter(data, [device])[0]
        if ref:
            data['data_samples'][0].img_lq.data = data['data_samples'][
                0].img_lq.data.to(device)
            data['data_samples'][0].ref_lq.data = data['data_samples'][
                0].ref_lq.data.to(device)
            data['data_samples'][0].ref_img.data = data['data_samples'][
                0].ref_img.data.to(device)
    # forward the model
    with torch.no_grad():
        result = model(mode='tensor', **data)
    result = result[0]
    return result


try:
    from facexlib.utils.face_restoration_helper import FaceRestoreHelper
    has_facexlib = True
except ImportError:
    has_facexlib = False


def restoration_face_inference(model, img, upscale_factor=1, face_size=1024):
    """Inference image with the model.

    Args:
        model (nn.Module): The loaded model.
        img (str): File path of input image.
        upscale_factor (int, optional): The number of times the input image
            is upsampled. Default: 1.
        face_size (int, optional): The size of the cropped and aligned faces.
            Default: 1024.

    Returns:
        Tensor: The predicted restoration result.
    """
    device = next(model.parameters()).device  # model device

    # build the data pipeline
    if model.cfg.get('demo_pipeline', None):
        test_pipeline = model.cfg.demo_pipeline
    elif model.cfg.get('test_pipeline', None):
        test_pipeline = model.cfg.test_pipeline
    else:
        test_pipeline = model.cfg.val_pipeline

    # remove gt from test_pipeline
    keys_to_remove = ['gt', 'gt_path']
    for key in keys_to_remove:
        for pipeline in list(test_pipeline):
            if 'key' in pipeline and key == pipeline['key']:
                test_pipeline.remove(pipeline)
            if 'keys' in pipeline and key in pipeline['keys']:
                pipeline['keys'].remove(key)
                if len(pipeline['keys']) == 0:
                    test_pipeline.remove(pipeline)
            if 'meta_keys' in pipeline and key in pipeline['meta_keys']:
                pipeline['meta_keys'].remove(key)
    # build the data pipeline
    test_pipeline = Compose(test_pipeline)

    # face helper for detecting and aligning faces
    assert has_facexlib, 'Please install FaceXLib to use the demo.'
    face_helper = FaceRestoreHelper(
        upscale_factor,
        face_size=face_size,
        crop_ratio=(1, 1),
        det_model='retinaface_resnet50',
        template_3points=True,
        save_ext='png',
        device=device)

    face_helper.read_image(img)
    # get face landmarks for each face
    face_helper.get_face_landmarks_5(
        only_center_face=False, eye_dist_threshold=None)
    # align and warp each face
    face_helper.align_warp_face()

    for i, img in enumerate(face_helper.cropped_faces):
        # prepare data
        mmcv.imwrite(img, 'demo/tmp.png')
        data = dict(lq=img.astype(np.float32), img_path='demo/tmp.png')
        _data = test_pipeline(data)
        data = dict()
        data['inputs'] = _data['inputs'] / 255.0
        data = collate([data])
        if 'cuda' in str(device):
            data = scatter(data, [device])[0]

        with torch.no_grad():
            output = model(mode='tensor', **data)

        output = output.squeeze(0).permute(1, 2, 0)[:, :, [2, 1, 0]]
        output = output.cpu().numpy() * 255  # (0, 255)
        face_helper.add_restored_face(output)

    face_helper.get_inverse_affine(None)
    restored_img = face_helper.paste_faces_to_input_image(upsample_img=None)

    return restored_img


def pad_sequence(data, window_size):
    """Pad frame sequence data.

    Args:
        data (Tensor): The frame sequence data.
        window_size (int): The window size used in sliding-window framework.

    Returns:
        data (Tensor): The padded result.
    """

    padding = window_size // 2

    data = torch.cat([
        data[:, 1 + padding:1 + 2 * padding].flip(1), data,
        data[:, -1 - 2 * padding:-1 - padding].flip(1)
    ],
                     dim=1)

    return data


def restoration_video_inference(model,
                                img_dir,
                                window_size,
                                start_idx,
                                filename_tmpl,
                                max_seq_len=None):
    """Inference image with the model.

    Args:
        model (nn.Module): The loaded model.
        img_dir (str): Directory of the input video.
        window_size (int): The window size used in sliding-window framework.
            This value should be set according to the settings of the network.
            A value smaller than 0 means using recurrent framework.
        start_idx (int): The index corresponds to the first frame in the
            sequence.
        filename_tmpl (str): Template for file name.
        max_seq_len (int | None): The maximum sequence length that the model
            processes. If the sequence length is larger than this number,
            the sequence is split into multiple segments. If it is None,
            the entire sequence is processed at once.

    Returns:
        Tensor: The predicted restoration result.
    """

    device = next(model.parameters()).device  # model device

    # build the data pipeline
    if model.cfg.get('demo_pipeline', None):
        test_pipeline = model.cfg.demo_pipeline
    elif model.cfg.get('test_pipeline', None):
        test_pipeline = model.cfg.test_pipeline
    else:
        test_pipeline = model.cfg.val_pipeline

    # check if the input is a video
    file_extension = osp.splitext(img_dir)[1]
    if file_extension in VIDEO_EXTENSIONS:
        video_reader = mmcv.VideoReader(img_dir)
        # load the images
        data = dict(img=[], img_path=None, key=img_dir)
        for frame in video_reader:
            data['img'].append(np.flip(frame, axis=2))

        # remove the data loading pipeline
        tmp_pipeline = []
        for pipeline in test_pipeline:
            if pipeline['type'] not in [
                    'GenerateSegmentIndices', 'LoadImageFromFile'
            ]:
                tmp_pipeline.append(pipeline)
        test_pipeline = tmp_pipeline
    else:
        # the first element in the pipeline must be 'GenerateSegmentIndices'
        if test_pipeline[0]['type'] != 'GenerateSegmentIndices':
            raise TypeError('The first element in the pipeline must be '
                            f'"GenerateSegmentIndices", but got '
                            f'"{test_pipeline[0]["type"]}".')

        # specify start_idx and filename_tmpl
        test_pipeline[0]['start_idx'] = start_idx
        test_pipeline[0]['filename_tmpl'] = filename_tmpl

        # prepare data
        sequence_length = len(glob.glob(osp.join(img_dir, '*')))
        lq_folder = osp.dirname(img_dir)
        key = osp.basename(img_dir)
        data = dict(
            img_path=lq_folder,
            gt_path='',
            key=key,
            sequence_length=sequence_length)

    # compose the pipeline
    test_pipeline = Compose(test_pipeline)
    data = test_pipeline(data)
    data = data['inputs'].unsqueeze(0) / 255.0  # in cpu

    # forward the model
    with torch.no_grad():
        if window_size > 0:  # sliding window framework
            data = pad_sequence(data, window_size)
            result = []
            for i in range(0, data.size(1) - 2 * (window_size // 2)):
                data_i = data[:, i:i + window_size].to(device)
                result.append(model(inputs=data_i, mode='tensor').cpu())
            result = torch.stack(result, dim=1)
        else:  # recurrent framework
            if max_seq_len is None:
                result = model(inputs=data.to(device), mode='tensor').cpu()
            else:
                result = []
                for i in range(0, data.size(1), max_seq_len):
                    result.append(
                        model(
                            inputs=data[:, i:i + max_seq_len].to(device),
                            mode='tensor').cpu())
                result = torch.cat(result, dim=1)
    return result


def read_image(filepath):
    """Read image from file.

    Args:
        filepath (str): File path.

    Returns:
        image (np.array): Image.
    """
    img_bytes = FILE_CLIENT.get(filepath)
    image = mmcv.imfrombytes(
        img_bytes, flag='color', channel_order='rgb', backend='pillow')
    return image


def read_frames(source, start_index, num_frames, from_video, end_index):
    """Read frames from file or video.

    Args:
        source (list | mmcv.VideoReader): Source of frames.
        start_index (int): Start index of frames.
        num_frames (int): frames number to be read.
        from_video (bool): Weather read frames from video.
        end_index (int): The end index of frames.

    Returns:
        images (np.array): Images.
    """
    images = []
    last_index = min(start_index + num_frames, end_index)
    # read frames from video
    if from_video:
        for index in range(start_index, last_index):
            if index >= source.frame_cnt:
                break
            images.append(np.flip(source.get_frame(index), axis=2))
    else:
        files = source[start_index:last_index]
        images = [read_image(f) for f in files]
    return images


def video_interpolation_inference(model,
                                  input_dir,
                                  output_dir,
                                  start_idx=0,
                                  end_idx=None,
                                  batch_size=4,
                                  fps_multiplier=0,
                                  fps=0,
                                  filename_tmpl='{:08d}.png'):
    """Inference image with the model.

    Args:
        model (nn.Module): The loaded model.
        input_dir (str): Directory of the input video.
        output_dir (str): Directory of the output video.
        start_idx (int): The index corresponding to the first frame in the
            sequence. Default: 0
        end_idx (int | None): The index corresponding to the last interpolated
            frame in the sequence. If it is None, interpolate to the last
            frame of video or sequence. Default: None
        batch_size (int): Batch size. Default: 4
        fps_multiplier (float): multiply the fps based on the input video.
            Default: 0.
        fps (float): frame rate of the output video. Default: 0.
        filename_tmpl (str): template of the file names. Default: '{:08d}.png'
    """

    device = next(model.parameters()).device  # model device

    # build the data pipeline
    if model.cfg.get('demo_pipeline', None):
        test_pipeline = model.cfg.demo_pipeline
    elif model.cfg.get('test_pipeline', None):
        test_pipeline = model.cfg.test_pipeline
    else:
        test_pipeline = model.cfg.val_pipeline

    # remove the data loading pipeline
    tmp_pipeline = []
    for pipeline in test_pipeline:
        if pipeline['type'] not in [
                'GenerateSegmentIndices', 'LoadImageFromFile'
        ]:
            tmp_pipeline.append(pipeline)
    test_pipeline = tmp_pipeline

    # compose the pipeline
    test_pipeline = Compose(test_pipeline)

    # check if the input is a video
    input_file_extension = os.path.splitext(input_dir)[1]
    if input_file_extension in VIDEO_EXTENSIONS:
        source = mmcv.VideoReader(input_dir)
        input_fps = source.fps
        length = source.frame_cnt
        from_video = True
        h, w = source.height, source.width
        if fps_multiplier:
            assert fps_multiplier > 0, '`fps_multiplier` cannot be negative'
            output_fps = fps_multiplier * input_fps
        else:
            output_fps = fps if fps > 0 else input_fps * 2
    else:
        files = os.listdir(input_dir)
        files = [osp.join(input_dir, f) for f in files]
        files.sort()
        source = files
        length = files.__len__()
        from_video = False
        example_frame = read_image(files[0])
        h, w = example_frame.shape[:2]
        output_fps = fps

    # check if the output is a video
    output_file_extension = os.path.splitext(output_dir)[1]
    if output_file_extension in VIDEO_EXTENSIONS:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        target = cv2.VideoWriter(output_dir, fourcc, output_fps, (w, h))
        to_video = True
    else:
        to_video = False

    end_idx = min(end_idx, length) if end_idx is not None else length

    # calculate step args
    step_size = model.step_frames * batch_size
    lenth_per_step = model.required_frames + model.step_frames * (
        batch_size - 1)
    repeat_frame = model.required_frames - model.step_frames

    prog_bar = ProgressBar(
        math.ceil(
            (end_idx + step_size - lenth_per_step - start_idx) / step_size))
    output_index = start_idx
    for start_index in range(start_idx, end_idx, step_size):
        images = read_frames(
            source, start_index, lenth_per_step, from_video, end_index=end_idx)

        # data prepare
        data = dict(img=images, inputs_path=None, key=input_dir)
        data = test_pipeline(data)['inputs'] / 255.0
        data = collate([data])
        # data.shape: [1, t, c, h, w]

        # forward the model
        data = model.split_frames(data)
        input_tensors = data.clone().detach()
        with torch.no_grad():
            output = model(data.to(device), mode='tensor')
            if len(output.shape) == 4:
                output = output.unsqueeze(1)
            output_tensors = output.cpu()
            if len(output_tensors.shape) == 4:
                output_tensors = output_tensors.unsqueeze(1)
            result = model.merge_frames(input_tensors, output_tensors)
        if not start_idx == start_index:
            result = result[repeat_frame:]
        prog_bar.update()

        # save frames
        if to_video:
            for frame in result:
                target.write(frame)
        else:
            for frame in result:
                save_path = osp.join(output_dir,
                                     filename_tmpl.format(output_index))
                mmcv.imwrite(frame, save_path)
                output_index += 1

        if start_index + lenth_per_step >= end_idx:
            break

    print()
    print(f'Output dir: {output_dir}')
    if to_video:
        target.release()


def colorization_inference(model, img):
    """Inference image with the model.

    Args:
        model (nn.Module): The loaded model.
        img (str): Image file path.

    Returns:
        Tensor: The predicted colorization result.
    """
    device = next(model.parameters()).device

    # build the data pipeline
    test_pipeline = Compose(model.cfg.test_pipeline)
    # prepare data
    data = dict(img_path=img)
    _data = test_pipeline(data)
    data = dict()
    data['inputs'] = _data['inputs'] / 255.0
    data = collate([data])
    data['data_samples'] = [_data['data_samples']]
    if 'cuda' in str(device):
        data = scatter(data, [device])[0]
        if not data['data_samples'][0].empty_box:
            data['data_samples'][0].cropped_img.data = scatter(
                data['data_samples'][0].cropped_img.data, [device])[0] / 255.0

            data['data_samples'][0].box_info.data = scatter(
                data['data_samples'][0].box_info.data, [device])[0]

            data['data_samples'][0].box_info_2x.data = scatter(
                data['data_samples'][0].box_info_2x.data, [device])[0]

            data['data_samples'][0].box_info_4x.data = scatter(
                data['data_samples'][0].box_info_4x.data, [device])[0]

            data['data_samples'][0].box_info_8x.data = scatter(
                data['data_samples'][0].box_info_8x.data, [device])[0]

    # forward the model
    with torch.no_grad():
        result = model(mode='tensor', **data)

    return result


def calculate_grid_size(num_batches: int = 1, aspect_ratio: int = 1) -> int:
    """Calculate the number of images per row (nrow) to make the grid closer to
    square when formatting a batch of images to grid.

    Args:
        num_batches (int, optional): Number of images per batch. Defaults to 1.
        aspect_ratio (int, optional): The aspect ratio (width / height) of
            each image sample. Defaults to 1.

    Returns:
        int: Calculated number of images per row.
    """
    curr_ncol, curr_nrow = 1, num_batches
    curr_delta = curr_nrow * aspect_ratio - curr_ncol

    nrow = curr_nrow
    delta = curr_delta

    while curr_delta > 0:

        curr_ncol += 1
        curr_nrow = math.ceil(num_batches / curr_ncol)

        curr_delta = curr_nrow * aspect_ratio - curr_ncol
        if curr_delta < delta and curr_delta >= 0:
            nrow, delta = curr_nrow, curr_delta

    return nrow
