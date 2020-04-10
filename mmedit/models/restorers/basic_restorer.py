from ..base import BaseModel
from ..builder import build_backbone, build_loss
from ..registry import MODELS


@MODELS.register_module
class BasicRestorer(BaseModel):
    """Basic model for image restoration.

    It must contain a generator that takes an image as inputs and outputs a
    restored image. It also has a pixel-wise loss for training.

    The subclasses should overwrite the function `forward_train`,
    `forward_test` and `train_step`.

    Args:
        generator (dict): Config for the generator structure.
        pixel_loss (dict): Config for pixel-wise loss.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path for pretrained model. Default: None.
    """

    def __init__(self,
                 generator,
                 pixel_loss,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(BasicRestorer, self).__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # generator
        self.generator = build_backbone(generator)
        self.init_weights(pretrained)

        # loss
        self.pixel_loss = build_loss(pixel_loss)

    def init_weights(self, pretrained=None):
        self.generator.init_weights(pretrained)

    def forward(self, lq, gt=None, test_mode=False, **kwargs):
        if not test_mode:
            return self.forward_train(lq, gt)
        else:
            return self.forward_test(lq)

    def forward_train(self, lq, gt):
        losses = dict()
        output = self.generator(lq)
        loss_pix = self.pixel_loss(output, gt)
        losses['loss_pix'] = loss_pix
        outputs = dict(
            losses=losses,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), gt=gt.cpu(), output=output.cpu()))
        return outputs

    def forward_test(self, lq):
        output = self.generator(lq)
        results = dict(lq=lq.cpu(), output=output.cpu())
        return results

    def forward_dummy(self, img):
        """Used for computing network flops."""
        out = self.generator(img)
        return out

    def train_step(self, data_batch, optimizer):
        outputs = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(outputs.pop('losses'))

        # optimize
        optimizer['generator'].zero_grad()
        loss.backward()
        optimizer['generator'].step()

        outputs.update({'log_vars': log_vars})
        return outputs

    def val_step(self, data_batch):
        output = self.forward_test(data_batch['lq'])
        return output
