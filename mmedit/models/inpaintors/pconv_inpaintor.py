from ..registry import MODELS
from .one_stage import OneStageInpaintor


@MODELS.register_module
class PConvInpaintor(OneStageInpaintor):

    def forward_test(self, masked_img, mask, **kwargs):
        mask_input = mask.expand_as(masked_img)
        mask_input = 1. - mask_input

        fake_res, final_mask = self.generator(masked_img, mask_input)
        fake_img = fake_res * mask + masked_img * (1. - mask)
        output = dict(
            fake_res=fake_res, fake_img=fake_img, final_mask=final_mask)
        return output

    def train_step(self, data_batch, optimizer):
        log_vars = {}

        gt_img = data_batch['gt_img']
        mask = data_batch['mask']
        masked_img = data_batch['masked_img']

        mask_input = mask.expand_as(gt_img)
        mask_input = 1. - mask_input

        fake_res, final_mask = self.generator(masked_img, mask_input)
        fake_img = gt_img * (1. - mask) + fake_res * mask

        results, g_losses = self.generator_loss(fake_res, fake_img, data_batch)
        loss_g_, log_vars_g = self.parse_losses(g_losses)
        log_vars.update(log_vars_g)
        optimizer['generator'].zero_grad()
        loss_g_.backward()
        optimizer['generator'].step()

        results.update(dict(final_mask=final_mask))
        outputs = dict(
            log_vars=log_vars,
            num_samples=len(data_batch['gt_img'].data),
            results=results)

        return outputs
