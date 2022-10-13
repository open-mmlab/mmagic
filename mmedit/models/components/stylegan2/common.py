# Copyright (c) OpenMMLab. All rights reserved.
import torch


def get_module_device(module):
    """Get the device of a module.

    Args:
        module (nn.Module): A module contains the parameters.

    Returns:
        torch.device: The device of the module.
    """
    try:
        next(module.parameters())
    except StopIteration:
        raise ValueError('The input module should contain parameters.')

    if next(module.parameters()).is_cuda:
        return next(module.parameters()).get_device()
    else:
        return torch.device('cpu')


@torch.no_grad()
def get_mean_latent(generator, num_samples=4096, bs_per_repeat=1024):
    """Get mean latent of W space in Style-based GANs.

    Args:
        generator (nn.Module): Generator of a Style-based GAN.
        num_samples (int, optional): Number of sample times. Defaults to 4096.
        bs_per_repeat (int, optional): Batch size of noises per sample.
            Defaults to 1024.
    Returns:
        Tensor: Mean latent of this generator.
    """
    device = get_module_device(generator)
    mean_style = None
    n_repeat = num_samples // bs_per_repeat
    assert n_repeat * bs_per_repeat == num_samples

    for i in range(n_repeat):
        style = generator.style_mapping(
            torch.randn(bs_per_repeat,
                        generator.style_channels).to(device)).mean(
                            0, keepdim=True)
        if mean_style is None:
            mean_style = style
        else:
            mean_style += style
    mean_style /= float(n_repeat)

    return mean_style


@torch.no_grad()
def style_mixing(generator,
                 n_source,
                 n_target,
                 inject_index=1,
                 truncation_latent=None,
                 truncation=0.7,
                 style_channels=512,
                 **kwargs):
    device = get_module_device(generator)
    source_code = torch.randn(n_source, style_channels).to(device)
    target_code = torch.randn(n_target, style_channels).to(device)

    source_image = generator(
        source_code,
        truncation_latent=truncation_latent,
        truncation=truncation,
        **kwargs)

    h, w = source_image.shape[-2:]
    images = [torch.ones(1, 3, h, w).to(device) * -1]

    target_image = generator(
        target_code,
        truncation_latent=truncation_latent,
        truncation=truncation,
        **kwargs)

    images.append(source_image)

    for i in range(n_target):
        image = generator(
            [target_code[i].unsqueeze(0).repeat(n_source, 1), source_code],
            truncation_latent=truncation_latent,
            truncation=truncation,
            inject_index=inject_index,
            **kwargs)
        images.append(target_image[i].unsqueeze(0))
        images.append(image)

    images = torch.cat(images, 0)

    return images
