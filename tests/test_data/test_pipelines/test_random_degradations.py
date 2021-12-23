# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest

from mmedit.datasets.pipelines import (DegradationsWithShuffle, RandomBlur,
                                       RandomJPEGCompression, RandomNoise,
                                       RandomResize, RandomVideoCompression)


def test_random_noise():
    results = {}
    results['lq'] = np.ones((8, 8, 3)).astype(np.float32)

    # Gaussian noise
    model = RandomNoise(
        params=dict(
            noise_type=['gaussian'],
            noise_prob=[1],
            gaussian_sigma=[0, 50],
            gaussian_gray_noise_prob=1),
        keys=['lq'])
    results = model(results)
    assert results['lq'].shape == (8, 8, 3)

    # Poisson noise
    model = RandomNoise(
        params=dict(
            noise_type=['poisson'],
            noise_prob=[1],
            poisson_scale=[0, 1],
            poisson_gray_noise_prob=1),
        keys=['lq'])
    results = model(results)
    assert results['lq'].shape == (8, 8, 3)

    # skip degradations with prob < 1
    params = dict(
        noise_type=['gaussian'],
        noise_prob=[1],
        gaussian_sigma=[0, 50],
        gaussian_gray_noise_prob=1,
        prob=0)
    model = RandomNoise(params=params, keys=['lq'])
    assert model(results) == results

    assert repr(model) == model.__class__.__name__ + f'(params={params}, ' \
        + "keys=['lq'])"


def test_random_jpeg_compression():
    results = {}
    results['lq'] = np.ones((8, 8, 3)).astype(np.float32)

    model = RandomJPEGCompression(params=dict(quality=[5, 50]), keys=['lq'])
    results = model(results)
    assert results['lq'].shape == (8, 8, 3)

    # skip degradations with prob < 1
    params = dict(quality=[5, 50], prob=0)
    model = RandomJPEGCompression(params=params, keys=['lq'])
    assert model(results) == results

    assert repr(model) == model.__class__.__name__ + f'(params={params}, ' \
        + "keys=['lq'])"


def test_random_video_compression():
    results = {}
    results['lq'] = [np.ones((8, 8, 3)).astype(np.float32)] * 5

    model = RandomVideoCompression(
        params=dict(
            codec=['libx264', 'h264', 'mpeg4'],
            codec_prob=[1 / 3., 1 / 3., 1 / 3.],
            bitrate=[1e4, 1e5]),
        keys=['lq'])
    results = model(results)
    assert results['lq'][0].shape == (8, 8, 3)
    assert len(results['lq']) == 5

    # skip degradations with prob < 1
    params = dict(
        codec=['libx264', 'h264', 'mpeg4'],
        codec_prob=[1 / 3., 1 / 3., 1 / 3.],
        bitrate=[1e4, 1e5],
        prob=0)
    model = RandomVideoCompression(params=params, keys=['lq'])
    assert model(results) == results

    assert repr(model) == model.__class__.__name__ + f'(params={params}, ' \
        + "keys=['lq'])"


def test_random_resize():
    results = {}
    results['lq'] = np.ones((8, 8, 3)).astype(np.float32)

    # upscale
    model = RandomResize(
        params=dict(
            resize_mode_prob=[1, 0, 0],
            resize_scale=[0.5, 1.5],
            resize_opt=['bilinear', 'area', 'bicubic'],
            resize_prob=[1 / 3., 1 / 3., 1 / 3.]),
        keys=['lq'])
    results = model(results)
    assert results['lq'].shape[0] >= 8 and results['lq'].shape[1] >= 8

    # downscale
    results['lq'] = np.ones((8, 8, 3)).astype(np.float32)
    model = RandomResize(
        params=dict(
            resize_mode_prob=[0, 1, 0],
            resize_scale=[0.5, 1.5],
            resize_opt=['bilinear', 'area', 'bicubic'],
            resize_prob=[1 / 3., 1 / 3., 1 / 3.]),
        keys=['lq'])
    results = model(results)
    assert results['lq'].shape[0] <= 8 and results['lq'].shape[1] <= 8

    # keep size
    results['lq'] = np.ones((8, 8, 3)).astype(np.float32)
    model = RandomResize(
        params=dict(
            resize_mode_prob=[0, 0, 1],
            resize_scale=[0.5, 1.5],
            resize_opt=['bilinear', 'area', 'bicubic'],
            resize_prob=[1 / 3., 1 / 3., 1 / 3.]),
        keys=['lq'])
    results = model(results)
    assert results['lq'].shape[0] == 8 and results['lq'].shape[1] == 8

    # given target_size
    results['lq'] = np.ones((8, 8, 3)).astype(np.float32)
    model = RandomResize(
        params=dict(
            resize_mode_prob=[0, 0, 1],
            resize_scale=[0.5, 1.5],
            resize_opt=['bilinear', 'area', 'bicubic'],
            resize_prob=[1 / 3., 1 / 3., 1 / 3.],
            target_size=(16, 32)),
        keys=['lq'])
    results = model(results)
    assert results['lq'].shape == (16, 32, 3)

    # step_size > 0
    results['lq'] = np.ones((8, 8, 3)).astype(np.float32)
    model = RandomResize(
        params=dict(
            resize_mode_prob=[0, 0, 1],
            resize_scale=[0.5, 1.5],
            resize_opt=['bilinear', 'area', 'bicubic'],
            resize_prob=[1 / 3., 1 / 3., 1 / 3.],
            resize_step=0.05),
        keys=['lq'])
    results = model(results)

    # is_size_even is True
    results['lq'] = np.ones((8, 8, 3)).astype(np.float32)
    model = RandomResize(
        params=dict(
            resize_mode_prob=[0, 1, 0],
            resize_scale=[0.5, 1.5],
            resize_opt=['bilinear', 'area', 'bicubic'],
            resize_prob=[1 / 3., 1 / 3., 1 / 3.],
            resize_step=0.05,
            is_size_even=True),
        keys=['lq'])
    results = model(results)
    assert results['lq'].shape[0] % 2 == 0
    assert results['lq'].shape[1] % 2 == 0

    # skip degradation
    model = RandomResize(
        params=dict(
            resize_mode_prob=[1, 0, 0],
            resize_scale=[0.5, 1.5],
            resize_opt=['bilinear', 'area', 'bicubic'],
            resize_prob=[1 / 3., 1 / 3., 1 / 3.],
            prob=0),
        keys=['lq'])
    assert model(results) == results

    with pytest.raises(NotImplementedError):
        params = dict(
            resize_mode_prob=[1],
            resize_scale=[1],
            resize_opt=['abc'],
            resize_prob=[1])
        model = RandomResize(params=params, keys=['lq'])
        results = model(results)

    assert repr(model) == model.__class__.__name__ + f'(params={params}, ' \
        + "keys=['lq'])"


def test_random_blur():
    results = {}
    results['lq'] = np.ones((8, 8, 3)).astype(np.float32)

    # isotropic Gaussian
    model = RandomBlur(
        params=dict(
            kernel_size=[41],
            kernel_list=['iso'],
            kernel_prob=[1],
            sigma_x=[0.2, 10],
            sigma_y=[0.2, 10],
            rotate_angle=[-3.1416, 3.1416]),
        keys=['lq'])
    results = model(results)
    assert results['lq'].shape == (8, 8, 3)

    # anisotropic Gaussian
    model = RandomBlur(
        params=dict(
            kernel_size=[41],
            kernel_list=['aniso'],
            kernel_prob=[1],
            sigma_x=[0.2, 10],
            sigma_y=[0.2, 10],
            rotate_angle=[-3.1416, 3.1416]),
        keys=['lq'])
    results = model(results)
    assert results['lq'].shape == (8, 8, 3)

    # isotropic generalized Gaussian
    model = RandomBlur(
        params=dict(
            kernel_size=[41],
            kernel_list=['generalized_iso'],
            kernel_prob=[1],
            sigma_x=[0.2, 10],
            sigma_y=[0.2, 10],
            rotate_angle=[-3.1416, 3.1416]),
        keys=['lq'])
    results = model(results)
    assert results['lq'].shape == (8, 8, 3)

    # anisotropic generalized Gaussian
    model = RandomBlur(
        params=dict(
            kernel_size=[41],
            kernel_list=['generalized_aniso'],
            kernel_prob=[1],
            sigma_x=[0.2, 10],
            sigma_y=[0.2, 10],
            rotate_angle=[-3.1416, 3.1416]),
        keys=['lq'])
    results = model(results)
    assert results['lq'].shape == (8, 8, 3)

    # isotropic plateau Gaussian
    model = RandomBlur(
        params=dict(
            kernel_size=[41],
            kernel_list=['plateau_iso'],
            kernel_prob=[1],
            sigma_x=[0.2, 10],
            sigma_y=[0.2, 10],
            rotate_angle=[-3.1416, 3.1416]),
        keys=['lq'])
    results = model(results)
    assert results['lq'].shape == (8, 8, 3)

    # anisotropic plateau Gaussian
    model = RandomBlur(
        params=dict(
            kernel_size=[41],
            kernel_list=['plateau_aniso'],
            kernel_prob=[1],
            sigma_x=[0.2, 10],
            sigma_y=[0.2, 10],
            rotate_angle=[-3.1416, 3.1416]),
        keys=['lq'])
    results = model(results)
    assert results['lq'].shape == (8, 8, 3)

    # sinc (kernel size < 13)
    model = RandomBlur(
        params=dict(
            kernel_size=[11],
            kernel_list=['sinc'],
            kernel_prob=[1],
            sigma_x=[0.2, 10],
            sigma_y=[0.2, 10],
            rotate_angle=[-3.1416, 3.1416]),
        keys=['lq'])
    results = model(results)
    assert results['lq'].shape == (8, 8, 3)

    # sinc (kernel size >= 13)
    model = RandomBlur(
        params=dict(
            kernel_size=[15],
            kernel_list=['sinc'],
            kernel_prob=[1],
            sigma_x=[0.2, 10],
            sigma_y=[0.2, 10],
            rotate_angle=[-3.1416, 3.1416]),
        keys=['lq'])
    results = model(results)
    assert results['lq'].shape == (8, 8, 3)

    # sinc (given omega)
    model = RandomBlur(
        params=dict(
            kernel_size=[15],
            kernel_list=['sinc'],
            kernel_prob=[1],
            sigma_x=[0.2, 10],
            sigma_y=[0.2, 10],
            rotate_angle=[-3.1416, 3.1416],
            omega=[0.1, 0.1]),
        keys=['lq'])
    results = model(results)
    assert results['lq'].shape == (8, 8, 3)

    # skip degradation
    params = dict(
        kernel_size=[15],
        kernel_list=['sinc'],
        kernel_prob=[1],
        sigma_x=[0.2, 10],
        sigma_y=[0.2, 10],
        rotate_angle=[-3.1416, 3.1416],
        prob=0)
    model = RandomBlur(params=params, keys=['lq'])
    assert model(results) == results

    assert repr(model) == model.__class__.__name__ + f'(params={params}, ' \
        + "keys=['lq'])"


def test_degradations_with_shuffle():
    results = {}
    results['lq'] = np.ones((8, 8, 3)).astype(np.float32)

    # shuffle all
    model = DegradationsWithShuffle(
        degradations=[
            dict(
                type='RandomBlur',
                params=dict(
                    kernel_size=[15],
                    kernel_list=['sinc'],
                    kernel_prob=[1],
                    sigma_x=[0.2, 10],
                    sigma_y=[0.2, 10],
                    rotate_angle=[-3.1416, 3.1416],
                    omega=[0.1, 0.1])),
            dict(
                type='RandomResize',
                params=dict(
                    resize_mode_prob=[0, 0, 1],
                    resize_scale=[0.5, 1.5],
                    resize_opt=['bilinear', 'area', 'bicubic'],
                    resize_prob=[1 / 3., 1 / 3., 1 / 3.],
                    target_size=(16, 16))),
            [
                dict(
                    type='RandomJPEGCompression',
                    params=dict(quality=[5, 10])),
                dict(
                    type='RandomJPEGCompression',
                    params=dict(quality=[15, 20]))
            ]
        ],
        keys=['lq'],
        shuffle_idx=None)
    model(results)

    # shuffle last 2
    degradations = [
        dict(
            type='RandomBlur',
            params=dict(
                kernel_size=[15],
                kernel_list=['sinc'],
                kernel_prob=[1],
                sigma_x=[0.2, 10],
                sigma_y=[0.2, 10],
                rotate_angle=[-3.1416, 3.1416],
                omega=[0.1, 0.1])),
        dict(
            type='RandomResize',
            params=dict(
                resize_mode_prob=[0, 0, 1],
                resize_scale=[0.5, 1.5],
                resize_opt=['bilinear', 'area', 'bicubic'],
                resize_prob=[1 / 3., 1 / 3., 1 / 3.],
                target_size=(16, 16))),
        [
            dict(type='RandomJPEGCompression', params=dict(quality=[5, 10])),
            dict(type='RandomJPEGCompression', params=dict(quality=[15, 20]))
        ]
    ]
    model = DegradationsWithShuffle(
        degradations=degradations, keys=['lq'], shuffle_idx=(1, 2))
    model(results)

    assert repr(model) == model.__class__.__name__ \
        + f'(degradations={degradations}, ' \
        + "keys=['lq'], " \
        + 'shuffle_idx=(1, 2))'
