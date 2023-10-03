# Copyright (c) OpenMMLab. All rights reserved.
import math

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn


def device():
    """return torch.device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def get_weightings(weight_schedule, snrs, sigma_data):
    """return weightings."""
    if weight_schedule == 'snr':
        weightings = snrs
    elif weight_schedule == 'snr+1':
        weightings = snrs + 1
    elif weight_schedule == 'karras':
        weightings = snrs + 1.0 / sigma_data**2
    elif weight_schedule == 'truncated-snr':
        weightings = torch.clamp(snrs, min=1.0)
    elif weight_schedule == 'uniform':
        weightings = torch.ones_like(snrs)
    else:
        raise NotImplementedError()
    return weightings


class SiLU(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm32(nn.GroupNorm):

    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def karras_sample(
        diffusion,
        model,
        shape,
        steps,
        clip_denoised=True,
        progress=False,
        callback=None,
        model_kwargs=None,
        device=None,
        sigma_min=0.002,
        sigma_max=80,  # higher for highres?
        rho=7.0,
        sampler='heun',
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float('inf'),
        s_noise=1.0,
        generator=None,
        ts=None,
):
    if generator is None:
        generator = get_generator('dummy')

    if sampler == 'progdist':
        sigmas = get_sigmas_karras(
            steps + 1, sigma_min, sigma_max, rho, device=device)
    else:
        sigmas = get_sigmas_karras(
            steps, sigma_min, sigma_max, rho, device=device)

    x_T = generator.randn(*shape, device=device) * sigma_max

    sample_fn = get_sample_fn(sampler)

    if sampler in ['heun', 'dpm']:
        sampler_args = dict(
            s_churn=s_churn, s_tmin=s_tmin, s_tmax=s_tmax, s_noise=s_noise)
    elif sampler == 'multistep':
        sampler_args = dict(
            ts=ts,
            t_min=sigma_min,
            t_max=sigma_max,
            rho=diffusion.rho,
            steps=steps)
    else:
        sampler_args = {}

    def denoiser(x_t, sigma):
        _, denoised = diffusion.denoise(model, x_t, sigma, **model_kwargs)
        if clip_denoised:
            denoised = denoised.clamp(-1, 1)
        return denoised

    x_0 = sample_fn(
        denoiser,
        x_T,
        sigmas,
        generator,
        progress=progress,
        callback=callback,
        **sampler_args,
    )
    return x_0.clamp(-1, 1)


def get_sample_fn(sampler):
    return {
        'heun': sample_heun,
        'dpm': sample_dpm,
        'ancestral': sample_euler_ancestral,
        'onestep': sample_onestep,
        'progdist': sample_progdist,
        'euler': sample_euler,
        'multistep': stochastic_iterative_sampler,
    }[sampler]


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)


def get_ancestral_step(sigma_from, sigma_to):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) /
                sigma_from**2)**0.5
    sigma_down = (sigma_to**2 - sigma_up**2)**0.5
    return sigma_down, sigma_up


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device='cpu'):
    """Constructs the noise schedule of Karras et al.

    (2022).
    """
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min**(1 / rho)
    max_inv_rho = sigma_max**(1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho))**rho
    return append_zero(sigmas).to(device)


@torch.no_grad()
def sample_euler_ancestral(model,
                           x,
                           sigmas,
                           generator,
                           progress=False,
                           callback=None):
    """Ancestral sampling with Euler method steps."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        denoised = model(x, sigmas[i] * s_in)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1])
        if callback is not None:
            callback({
                'x': x,
                'i': i,
                'sigma': sigmas[i],
                'sigma_hat': sigmas[i],
                'denoised': denoised,
            })
        d = to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        x = x + generator.randn_like(x) * sigma_up
    return x


@torch.no_grad()
def sample_midpoint_ancestral(model,
                              x,
                              ts,
                              generator,
                              progress=False,
                              callback=None):
    """Ancestral sampling with midpoint method steps."""
    s_in = x.new_ones([x.shape[0]])
    step_size = 1 / len(ts)
    if progress:
        from tqdm.auto import tqdm

        ts = tqdm(ts)

    for tn in ts:
        dn = model(x, tn * s_in)
        dn_2 = model(x + (step_size / 2) * dn, (tn + step_size / 2) * s_in)
        x = x + step_size * dn_2
        if callback is not None:
            callback({'x': x, 'tn': tn, 'dn': dn, 'dn_2': dn_2})
    return x


@torch.no_grad()
def sample_heun(
        denoiser,
        x,
        sigmas,
        generator,
        progress=False,
        callback=None,
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float('inf'),
        s_noise=1.0,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al.

    (2022).
    """
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 -
                1) if s_tmin <= sigmas[i] <= s_tmax else 0.0)
        eps = generator.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i]**2)**0.5
        denoised = denoiser(x, sigma_hat * s_in)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({
                'x': x,
                'i': i,
                'sigma': sigmas[i],
                'sigma_hat': sigma_hat,
                'denoised': denoised,
            })
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            # Euler method
            x = x + d * dt
        else:
            # Heun's method
            x_2 = x + d * dt
            denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
    return x


@torch.no_grad()
def sample_euler(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al.

    (2022).
    """
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        sigma = sigmas[i]
        denoised = denoiser(x, sigma * s_in)
        d = to_d(x, sigma, denoised)
        if callback is not None:
            callback({
                'x': x,
                'i': i,
                'sigma': sigmas[i],
                'denoised': denoised,
            })
        dt = sigmas[i + 1] - sigma
        x = x + d * dt
    return x


@torch.no_grad()
def sample_dpm(
        denoiser,
        x,
        sigmas,
        generator,
        progress=False,
        callback=None,
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float('inf'),
        s_noise=1.0,
):
    """A sampler inspired by DPM-Solver-2 and Algorithm 2 from Karras et al.

    (2022).
    """
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 -
                1) if s_tmin <= sigmas[i] <= s_tmax else 0.0)
        eps = generator.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i]**2)**0.5
        denoised = denoiser(x, sigma_hat * s_in)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({
                'x': x,
                'i': i,
                'sigma': sigmas[i],
                'sigma_hat': sigma_hat,
                'denoised': denoised,
            })
        sigma_mid = ((sigma_hat**(1 / 3) + sigmas[i + 1]**(1 / 3)) / 2)**3
        dt_1 = sigma_mid - sigma_hat
        dt_2 = sigmas[i + 1] - sigma_hat
        x_2 = x + d * dt_1
        denoised_2 = denoiser(x_2, sigma_mid * s_in)
        d_2 = to_d(x_2, sigma_mid, denoised_2)
        x = x + d_2 * dt_2
    return x


@torch.no_grad()
def sample_onestep(
    distiller,
    x,
    sigmas,
    generator=None,
    progress=False,
    callback=None,
):
    """Single-step generation from a distilled model."""
    s_in = x.new_ones([x.shape[0]])
    return distiller(x, sigmas[0] * s_in)


@torch.no_grad()
def stochastic_iterative_sampler(
    distiller,
    x,
    sigmas,
    generator,
    ts,
    progress=False,
    callback=None,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
):
    t_max_rho = t_max**(1 / rho)
    t_min_rho = t_min**(1 / rho)
    s_in = x.new_ones([x.shape[0]])

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho))**rho
        x0 = distiller(x, t * s_in)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) *
                  (t_min_rho - t_max_rho))**rho
        next_t = np.clip(next_t, t_min, t_max)
        x = x0 + generator.randn_like(x) * np.sqrt(next_t**2 - t_min**2)

    return x


@torch.no_grad()
def sample_progdist(
    denoiser,
    x,
    sigmas,
    generator=None,
    progress=False,
    callback=None,
):
    s_in = x.new_ones([x.shape[0]])
    sigmas = sigmas[:-1]  # skip the zero sigma

    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        sigma = sigmas[i]
        denoised = denoiser(x, sigma * s_in)
        d = to_d(x, sigma, denoised)
        if callback is not None:
            callback({
                'x': x,
                'i': i,
                'sigma': sigma,
                'denoised': denoised,
            })
        dt = sigmas[i + 1] - sigma
        x = x + d * dt

    return x


def conv_nd(dims, *args, **kwargs):
    """Create a 1D, 2D, or 3D convolution module."""
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f'unsupported dimensions: {dims}')


def linear(*args, **kwargs):
    """Create a linear module."""
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """Create a 1D, 2D, or 3D average pooling module."""
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f'unsupported dimensions: {dims}')


def update_ema(target_params, source_params, rate=0.99):
    """Update target parameters to be closer to those of source parameters
    using an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """Scale the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """Take the mean over all non-batch dimensions."""
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims
    dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims '
                         f'but target_dims is {target_dims}, which is less')
    return x[(..., ) + (None, ) * dims_to_append]


def append_zero(x):
    """add zeors."""
    return torch.cat([x, x.new_zeros([1])])


def normalization(channels):
    """Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) *
                      torch.arange(start=0, end=half, dtype=torch.float32) /
                      half).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding,
                               torch.zeros_like(embedding[:, :1])],
                              dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    """Evaluate a function without caching intermediate activations, allowing
    for reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [
            x.detach().requires_grad_(True) for x in ctx.input_tensors
        ]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def convert_module_to_f16(l1):
    """Convert primitive modules to float16."""
    if isinstance(l1, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l1.weight.data = l1.weight.data.half()
        if l1.bias is not None:
            l1.bias.data = l1.bias.data.half()


def convert_module_to_f32(l2):
    """Convert primitive modules to float32, undoing
    convert_module_to_f16()."""
    if isinstance(l2, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l2.weight.data = l2.weight.data.float()
        if l2.bias is not None:
            l2.bias.data = l2.bias.data.float()


def get_generator(generator, num_samples=0, seed=0):
    """return generator."""
    if generator == 'dummy':
        return DummyGenerator()
    elif generator == 'determ':
        return DeterministicGenerator(num_samples, seed)
    elif generator == 'determ-indiv':
        return DeterministicIndividualGenerator(num_samples, seed)
    else:
        raise NotImplementedError


class DummyGenerator:
    """return Dummy generator."""

    def randn(self, *args, **kwargs):
        """return random tensor."""
        return torch.randn(*args, **kwargs)

    def randint(self, *args, **kwargs):
        """return random int tensor."""
        return torch.randint(*args, **kwargs)

    def randn_like(self, *args, **kwargs):
        """return random like tensor."""
        return torch.randn_like(*args, **kwargs)


class DeterministicGenerator:
    """RNG to deterministically sample num_samples samples that does not depend
    on batch_size or mpi_machines Uses a single rng and samples num_samples
    sized randomness and subsamples the current indices."""

    def __init__(self, num_samples, seed=0):
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            print('Warning: Distributed not initialised, using single rank')
            self.rank = 0
            self.world_size = 1
        self.num_samples = num_samples
        self.done_samples = 0
        self.seed = seed
        self.rng_cpu = torch.Generator()
        if torch.cuda.is_available():
            self.rng_cuda = torch.Generator(device())
        self.set_seed(seed)

    def get_global_size_and_indices(self, size):
        """return size and indices."""
        global_size = (self.num_samples, *size[1:])
        indices = torch.arange(
            self.done_samples + self.rank,
            self.done_samples + self.world_size * int(size[0]),
            self.world_size,
        )
        indices = torch.clamp(indices, 0, self.num_samples - 1)
        assert (
            len(indices) == size[0]
        ), f'rank={self.rank}, ws={self.world_size}, ' \
           f'l={len(indices)}, bs={size[0]}'
        return global_size, indices

    def get_generator(self, device):
        """return rng generator."""
        return self.rng_cpu if torch.device(
            device).type == 'cpu' else self.rng_cuda

    def randn(self, *size, dtype=torch.float, device='cpu'):
        """return random tensor."""
        global_size, indices = self.get_global_size_and_indices(size)
        generator = self.get_generator(device)
        return torch.randn(
            *global_size, generator=generator, dtype=dtype,
            device=device)[indices]

    def randint(self, low, high, size, dtype=torch.long, device='cpu'):
        """return random int tensor."""
        global_size, indices = self.get_global_size_and_indices(size)
        generator = self.get_generator(device)
        return torch.randint(
            low,
            high,
            generator=generator,
            size=global_size,
            dtype=dtype,
            device=device)[indices]

    def randn_like(self, tensor):
        """return random like tensor."""
        size, dtype, device = tensor.size(), tensor.dtype, tensor.device
        return self.randn(*size, dtype=dtype, device=device)

    def set_done_samples(self, done_samples):
        """set model's done_samples."""
        self.done_samples = done_samples
        self.set_seed(self.seed)

    def get_seed(self):
        """return model's seed."""
        return self.seed

    def set_seed(self, seed):
        """set model's seed."""
        self.rng_cpu.manual_seed(seed)
        if torch.cuda.is_available():
            self.rng_cuda.manual_seed(seed)


class DeterministicIndividualGenerator:
    """RNG to deterministically sample num_samples samples that does not depend
    on batch_size or mpi_machines Uses a separate rng for each sample to reduce
    memory usage."""

    def __init__(self, num_samples, seed=0):
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            print('Warning: Distributed not initialised, using single rank')
            self.rank = 0
            self.world_size = 1
        self.num_samples = num_samples
        self.done_samples = 0
        self.seed = seed
        self.rng_cpu = [torch.Generator() for _ in range(num_samples)]
        if torch.cuda.is_available():
            self.rng_cuda = [
                torch.Generator(device()) for _ in range(num_samples)
            ]
        self.set_seed(seed)

    def get_size_and_indices(self, size):
        """return size and indices."""
        indices = torch.arange(
            self.done_samples + self.rank,
            self.done_samples + self.world_size * int(size[0]),
            self.world_size,
        )
        indices = torch.clamp(indices, 0, self.num_samples - 1)
        assert (
            len(indices) == size[0]
        ), f'rank={self.rank}, ws={self.world_size}, ' \
           f'l={len(indices)}, bs={size[0]}'
        return (1, *size[1:]), indices

    def get_generator(self, device):
        """return generator."""
        return self.rng_cpu if torch.device(
            device).type == 'cpu' else self.rng_cuda

    def randn(self, *size, dtype=torch.float, device='cpu'):
        """return random generator."""
        size, indices = self.get_size_and_indices(size)
        generator = self.get_generator(device)
        return torch.cat(
            [
                torch.randn(
                    *size, generator=generator[i], dtype=dtype, device=device)
                for i in indices
            ],
            dim=0,
        )

    def randint(self, low, high, size, dtype=torch.long, device='cpu'):
        """return random int generator."""
        size, indices = self.get_size_and_indices(size)
        generator = self.get_generator(device)
        return torch.cat(
            [
                torch.randint(
                    low,
                    high,
                    generator=generator[i],
                    size=size,
                    dtype=dtype,
                    device=device,
                ) for i in indices
            ],
            dim=0,
        )

    def randn_like(self, tensor):
        """return random like tensor."""
        size, dtype, device = tensor.size(), tensor.dtype, tensor.device
        return self.randn(*size, dtype=dtype, device=device)

    def set_done_samples(self, done_samples):
        """set model's done_samples."""
        self.done_samples = done_samples

    def get_seed(self):
        """return model's seed."""
        return self.seed

    def set_seed(self, seed):
        """set model's seed."""
        [
            rng_cpu.manual_seed(i + self.num_samples * seed)
            for i, rng_cpu in enumerate(self.rng_cpu)
        ]
        if torch.cuda.is_available():
            [
                rng_cuda.manual_seed(i + self.num_samples * seed)
                for i, rng_cuda in enumerate(self.rng_cuda)
            ]
