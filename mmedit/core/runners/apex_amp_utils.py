try:
    from apex import amp
except ImportError:
    amp = None


def apex_amp_initialize(models, optimizers, init_args=None, mode='gan'):
    """Initialize apex.amp for mixed-precision training.
    Args:
        models (nn.Module | list[Module]): Modules to be wrapped with apex.amp.
        optimizer (:obj:`Optimizer`, optional): Optimizer to be saved.
        init_args (dict | None, optional): Config for amp initialization.
            Defaults to None.
        mode (str, optional): The moded used to initialize the apex.map.
            Different modes lead to different wrapping mode for models and
            optimizers. Defaults to 'gan'.
    Returns:
        Module, :obj:`Optimizer`: Wrapped module and optimizer.
    """
    init_args = init_args or dict()

    if mode == 'gan':
        _optmizers = [optimizers['generator'], optimizers['discriminator']]

        models, _optmizers = amp.initialize(models, _optmizers, **init_args)
        optimizers['generator'] = _optmizers[0]
        optimizers['discriminator'] = _optmizers[1]

        return models, optimizers

    else:
        raise NotImplementedError(
            f'Cannot initialize apex.amp with mode {mode}')
