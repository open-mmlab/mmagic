from mmagic.models import ResNet
from mmagic.registry import MODELS


# Register your model to the `MODELS`.
@MODELS.register_module()
class ExampleNet(ResNet):
    """Implements an example backbone.

    Implement the backbone network just like a normal pytorch network.
    """

    def __init__(self, **kwargs) -> None:
        print('#############################\n'
              '#  Hello MMagic!  #\n'
              '#############################')
        super().__init__(**kwargs)

    def forward(self, x):
        """The forward method of the network.

        Args:
            x (torch.Tensor): A tensor of image batch with shape
                ``(batch_size, num_channels, height, width)``.

        Returns:
            Tuple[torch.Tensor]: Please return a tuple of tensors and every
            tensor is a feature map of specified scale. If you only want the
            final feature map, simply return a tuple with one item.
        """
        return super().forward(x)
