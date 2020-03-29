import torch.nn as nn

pad_cfg = {
    'reflect': nn.ReflectionPad2d,
}


def build_padding_layer(cfg, *args, **kwargs):
    """ Build padding layer

    Args:
        cfg (None or dict): cfg should contain:
            type (str): identify padding layer type.
            layer args: args needed to instantiate a padding layer.

    Returns:
        nn.Module: created padding layer
    """

    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()
    padding_type = cfg_.pop('type')
    if padding_type not in pad_cfg:
        raise KeyError('Unrecognized padding type {}'.format(padding_type))
    else:
        padding_layer = pad_cfg[padding_type]

    layer = padding_layer(*args, **kwargs, **cfg_)

    return layer
