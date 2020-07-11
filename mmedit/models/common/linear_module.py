import torch.nn as nn
from mmcv.cnn import build_activation_layer, kaiming_init


class LinearModule(nn.Module):
    """A linear block that contains linear/norm/activation layers.

    For low level visioin, we add spectral norm and padding layer.

    Args:
        in_features (int): Same as nn.Linear.
        out_features (int): Same as nn.Linear.
        bias (bool): Same as nn.Linear.
        act_cfg (dict): Config dict for activation layer, "relu" by default.
        inplace (bool): Whether to use inplace mode for activation.
        with_spectral_norm (bool): Whether use spectral norm in linear module.
        order (tuple[str]): The order of linear/activation layers. It is a
            sequence of "linear", "norm" and "act". Examples are
            ("linear", "act") and ("act", "linear").
    """

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 act_cfg=dict(type='ReLU'),
                 inplace=True,
                 with_spectral_norm=False,
                 order=('linear', 'act')):
        super(LinearModule, self).__init__()
        assert act_cfg is None or isinstance(act_cfg, dict)
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 2
        assert set(order) == set(['linear', 'act'])

        self.with_activation = act_cfg is not None
        self.with_bias = bias

        # build linear layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # export the attributes of self.linear to a higher level for
        # convenience
        self.in_features = self.linear.in_features
        self.out_features = self.linear.out_features

        if self.with_spectral_norm:
            self.linear = nn.utils.spectral_norm(self.linear)

        # build activation layer
        if self.with_activation:
            act_cfg_ = act_cfg.copy()
            act_cfg_.setdefault('inplace', inplace)
            self.activate = build_activation_layer(act_cfg_)

        # Use msra init by default
        self.init_weights()

    def init_weights(self):
        if self.with_activation and self.act_cfg['type'] == 'LeakyReLU':
            nonlinearity = 'leaky_relu'
            a = self.act_cfg.get('negative_slope', 0.01)
        else:
            nonlinearity = 'relu'
            a = 0

        kaiming_init(self.linear, a=a, nonlinearity=nonlinearity)

    def forward(self, x, activate=True):
        """Foward Function.

        Args:
            x (torch.Tensor): Input tensor with shape of (n, \*,  # noqa: W605
                c). Same as ``torch.nn.Linear``.
            activate (bool, optional): Whether to use activation layer.
                Defaults to True.

        Returns:
            torch.Tensor: Same as ``torch.nn.Linear``.
        """
        for layer in self.order:
            if layer == 'linear':
                x = self.linear(x)
            elif layer == 'act' and activate and self.with_activation:
                x = self.activate(x)
        return x
