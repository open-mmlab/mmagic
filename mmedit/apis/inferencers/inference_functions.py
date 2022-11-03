# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine import is_list_of

from mmengine import Config
from mmengine.config import ConfigDict
from mmengine.runner import load_checkpoint
from mmengine.runner import set_random_seed as set_random_seed_engine

from mmedit.registry import MODELS
from mmedit.utils import register_all_modules

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
    set_random_seed_engine(
        seed, deterministic=deterministic, use_rank_shift=use_rank_shift)


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
    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    # build the data pipeline
    infer_pipeline = [
                    dict(type='LoadImageFromFile', key='gt', channel_order='bgr'),
                    dict(
                        type='LoadMask',
                        mask_mode='file',
                    ),
                    dict(type='GetMaskedImage'),
                    dict(type='PackEditInputs'),]

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
            data['data_samples'][0].mask.data, [device])[0]
    # else:
    #     data.pop('meta')
    # forward the model
    with torch.no_grad():
        result, x = model(mode='tensor', **data)

    masks = _data['data_samples'].mask.data
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
