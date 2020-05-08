from ..registry import MODELS
from .basic_restorer import BasicRestorer


@MODELS.register_module
class EDVR(BasicRestorer):
    """EDVR model for video super-resolution.

    EDVR: Video Restoration with Enhanced Deformable Convolutional Networks.

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
        super(EDVR, self).__init__(generator, pixel_loss, train_cfg, test_cfg,
                                   pretrained)
        self.with_tsa = generator.get('with_tsa', False)
        self.step_counter = 0  # count training steps
        if self.with_tsa:
            if self.train_cfg is None or 'tsa_iter' not in self.train_cfg:
                raise KeyError(
                    'In TSA mode, train_cfg must contain "tsa_iter".')
            # only train TSA module at the beginging if with TSA module
            for k, v in self.generator.named_parameters():
                if 'fusion' not in k:
                    v.requires_grad = False

    def train_step(self, data_batch, optimizer):
        if self.with_tsa and (self.step_counter == self.train_cfg.tsa_iter):
            # train all the parameters
            for v in self.generator.parameters():
                v.requires_grad = True

        outputs = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(outputs.pop('losses'))

        # optimize
        optimizer['generator'].zero_grad()
        loss.backward()
        optimizer['generator'].step()

        self.step_counter += 1

        outputs.update({'log_vars': log_vars})
        return outputs
