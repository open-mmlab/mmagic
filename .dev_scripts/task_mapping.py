TASK_MAPPING = {
    'Inpainting':
    ['AOT-GAN', 'deepfillv1', 'deepfillv2', 'global_local', 'partial_conv'],
    'Matting': ['dim', 'gca', 'indexnet'],
    'Restoreration': [
        'srcnn', 'srresnet_srgan', 'edsr', 'esrgan', 'rdn', 'dic', 'ttsr',
        'glean', 'liif'
    ],
    'Video_restoration': [
        'edvr', 'tof', 'tdan', 'basicvsr', 'basicvsr_plusplus', 'iconvsr',
        'real_basicvsr'
    ],
    'Video_interpolation': ['cain', 'flavr', 'tof']
}
