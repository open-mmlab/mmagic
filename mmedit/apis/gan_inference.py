# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine import is_list_of


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
