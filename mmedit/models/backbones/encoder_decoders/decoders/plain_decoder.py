import torch.nn as nn
from mmcv.cnn.weight_init import xavier_init
from mmedit.models.registry import COMPONENTS


@COMPONENTS.register_module
class PlainDecoder(nn.Module):
    """Simple decoder from Deep Image Matting.

    Args:
        in_channels (int): Channel num of input features.
    """

    def __init__(self, in_channels):
        super(PlainDecoder, self).__init__()

        self.deconv6_1 = nn.Conv2d(in_channels, 512, kernel_size=1)
        self.deconv5_1 = nn.Conv2d(512, 512, kernel_size=5, padding=2)
        self.deconv4_1 = nn.Conv2d(512, 256, kernel_size=5, padding=2)
        self.deconv3_1 = nn.Conv2d(256, 128, kernel_size=5, padding=2)
        self.deconv2_1 = nn.Conv2d(128, 64, kernel_size=5, padding=2)
        self.deconv1_1 = nn.Conv2d(64, 64, kernel_size=5, padding=2)

        self.deconv1 = nn.Conv2d(64, 1, kernel_size=5, padding=2)

        self.relu = nn.ReLU(inplace=True)
        self.max_unpool2d = nn.MaxUnpool2d(kernel_size=2, stride=2)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m)

    def forward(self, x, max_indices):
        out = self.relu(self.deconv6_1(x))
        out = self.max_unpool2d(out, max_indices[4])

        out = self.relu(self.deconv5_1(out))
        out = self.max_unpool2d(out, max_indices[3])

        out = self.relu(self.deconv4_1(out))
        out = self.max_unpool2d(out, max_indices[2])

        out = self.relu(self.deconv3_1(out))
        out = self.max_unpool2d(out, max_indices[1])

        out = self.relu(self.deconv2_1(out))
        out = self.max_unpool2d(out, max_indices[0])

        out = self.relu(self.deconv1_1(out))
        raw_alpha = self.deconv1(out)
        return raw_alpha
