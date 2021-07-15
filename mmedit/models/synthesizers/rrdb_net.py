import os.path as osp

import mmcv
import numpy as np
import torch
from mmcv.runner import auto_fp16

from mmedit.core import tensor2img, psnr, ssim
from ..base import BaseModel
from ..builder import build_backbone, build_component, build_loss
from ..common import set_requires_grad
from ..registry import MODELS

@MODELS.register_module()
class RRDBnet(BaseModel):

    def __init__(self,
                 bgan_generator,
                 bgan_discriminator,
                 dbgan_generator,
                 dbgan_discriminator,
                 gan_loss,
                 perceptual_loss,
                 content_loss,
                 pixel_loss=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 is_useRBL=True):
        super().__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # generators
        self.bgan_generator = build_backbone(bgan_generator)
        self.dbgan_generator = build_backbone(dbgan_generator)
        
        # discrimiators
        self.bgan_discriminator = build_component(bgan_discriminator)
        self.dbgan_discriminator = build_component(dbgan_discriminator)


        # losses
        assert gan_loss is not None  # gan loss cannot be None
        
        self.use_RBLoss = is_useRBL
        
        self.bgan_loss = build_loss(gan_loss)
        self.dbgan_loss = build_loss(gan_loss)
        self.pixel_loss = build_loss(pixel_loss) if pixel_loss else None
        self.perceptual_loss = build_loss(perceptual_loss) if perceptual_loss else None
        self.content_loss = build_loss(content_loss) if content_loss else None
        
        self.disc_steps = 1 if self.train_cfg is None else self.train_cfg.get(
            'disc_steps', 1)
        self.disc_init_steps = (0 if self.train_cfg is None else
                                self.train_cfg.get('disc_init_steps', 0))
        
        self.step_counter = 0  # counting training steps

        self.show_input = (False if self.test_cfg is None else
                           self.test_cfg.get('show_input', False))

        # support fp16
        self.fp16_enabled = False
        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        """Initialize weights for the model.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Default: None.
        """
        self.bgan_generator.init_weights(pretrained=pretrained)
        self.bgan_discriminator.init_weights(pretrained=pretrained)
        self.dbgan_generator.init_weights(pretrained=pretrained)
        self.dbgan_discriminator.init_weights(pretrained=pretrained)

    def setup(self, real_sharp_a, real_blur_b, noise_map, meta):
        """Perform necessary pre-processing steps.

        Args:
            img_a (Tensor): Input image from domain A.
            img_b (Tensor): Input image from domain B.
            meta (list[dict]): Input meta data.

        Returns:
            Tensor, Tensor, list[str]: The real images from domain A/B, and \
                the image path as the metadata.
        """
        real_sharp_a = real_sharp_a
        real_blur_b = real_blur_b
        noise_map = noise_map
        return real_sharp_a, real_blur_b, noise_map, meta

    @auto_fp16(apply_to=('img_sharp_real', 'img_blur_real','noise_map'))
    def forward_train(self, img_sharp_real, img_blur_real, noise_map, meta):
        """Forward function for training.

        Args:
            img_a (Tensor): Input image from domain real sharp withNoise.
            img_b (Tensor): Input image from domain real blur(Rrdb).
            meta (list[dict]): Input meta data.

        Returns:
            dict: Dict of forward results for training.
        """
        # necessary setup
        real_sharp_a, real_blur_b, noise_map, meta = self.setup(img_sharp_real, img_blur_real,noise_map, meta)
        
        fake_blur_a = self.bgan_generator(real_sharp_a,noise_map) # 0 -1
        fake_sharp_a = self.dbgan_generator(fake_blur_a) # 0 -1

        results = dict(real_sharp_a = real_sharp_a,
                       real_blur_b  = real_blur_b, 
                       fake_blur_a = fake_blur_a,
                       fake_sharp_a = fake_sharp_a,
                       )
        return results

    def forward_test(self,
                     img_sharp_real,
                     img_blur_real,
                     noise_map,
                     ori_sharp_a,
                     meta = None,
                     save_image=1,
                     save_path=None,
                     iteration=None,
                     img_path=None,
                     **kwargs):
        """Forward function for testing.

        Args:
            img_a (Tensor): Input image from domain A.
            img_b (Tensor): Input image from domain B.
            meta (list[dict]): Input meta data.
            save_image (bool, optional): If True, results will be saved as
                images. Default: False.
            save_path (str, optional): If given a valid str path, the results
                will be saved in this path. Default: None.
            iteration (int, optional): Iteration number. Default: None.

        Returns:
            dict: Dict of forward and evaluation results for testing.
        """
        # No need for metrics during training for pix2pix. And
        # this is a special trick in pix2pix original paper & implementation,
        # collecting the statistics of the test batch at test time.
        self.eval()

        # necessary setup
        real_sharp_a, real_blur_a, noise_map, meta = self.setup( img_sharp_real, img_blur_real, noise_map, meta)
        
        fake_blur_a = self.bgan_generator(real_sharp_a,noise_map) # 0 -1
        fake_sharp_a_fromG = self.dbgan_generator(fake_blur_a) # 0 -1
        fake_sharp_a_fromR = self.dbgan_generator(real_blur_a)

        results=dict(fake_blur_a=fake_blur_a,
        fake_sharp_a_fromG=fake_sharp_a_fromG,
        fake_sharp_a_fromR=fake_sharp_a_fromR,
        real_sharp_a=real_sharp_a)

        img_fake_blur_a = tensor2img(results['fake_blur_a'], min_max=(-1,1))
        img_fake_sharp_a_fromG = tensor2img(results['fake_sharp_a_fromG'], min_max=(-1, 1))
        img_fake_sharp_a_fromR = tensor2img(results['fake_sharp_a_fromR'], min_max=(-1, 1)) 
        img_real_sharp_a = tensor2img(results['real_sharp_a'], min_max=(-1, 1)) 
        img_real_sharp_a_fromOri =ori_sharp_a.detach().cpu().numpy()
        
        img_blur_real_a = tensor2img(img_blur_real, min_max=(-1, 1)) 
        
        ret = {}
        self.show_input = True
        # save image
        img_path = meta[0]['img_sharp_real_path']
        if save_image:
            assert save_path is not None
            folder_name = osp.splitext(osp.basename(img_path))[0]
            if self.show_input:
                if iteration:
                    save_path = osp.join(
                        save_path, folder_name,
                        f'{folder_name}-{iteration + 1:06d}-rsa-fba-fsa.png')
                else:
                    save_path = osp.join(save_path,
                                         f'{folder_name}-rsa-fba-fsa.png')
                output = np.concatenate([
                   img_fake_blur_a,img_blur_real_a, img_fake_sharp_a_fromG, img_fake_sharp_a_fromR, img_real_sharp_a 
                ],
                                        axis=1)
            else:
                if iteration:
                    save_path = osp.join(
                        save_path, folder_name,
                        f'{folder_name}-{iteration + 1:06d}-fb.png')
                else:
                    save_path = osp.join(save_path, f'{folder_name}-fsa.png')
                output = tensor2img(results['fake_sharp_a_fromG'], min_max=(-1, 1))
            flag = mmcv.imwrite(output, save_path)
            ret['saved_flag'] = flag

        psnr_Gs_0S = psnr(img_fake_sharp_a_fromG, img_real_sharp_a_fromOri[0])
        psnr_Rs_0S = psnr(img_fake_sharp_a_fromR, img_real_sharp_a_fromOri[0])
        
        ssim_Gs_0S = ssim(img_fake_sharp_a_fromG, img_real_sharp_a_fromOri[0])
        ssim_Rs_0S = ssim(img_fake_sharp_a_fromR, img_real_sharp_a_fromOri[0])

        psnr_Gs_Ps = psnr(img_fake_sharp_a_fromG, img_real_sharp_a)
        psnr_Rs_Ps = psnr(img_fake_sharp_a_fromR, img_real_sharp_a)
        
        ssim_Gs_Ps = ssim(img_fake_sharp_a_fromG, img_real_sharp_a)
        ssim_Rs_Ps = ssim(img_fake_sharp_a_fromR, img_real_sharp_a)
        # print('--------------------------------------------------')
        # print(ssim(img_real_sharp_a,img_real_sharp_a_fromOri[0]))
        # print(psnr(img_real_sharp_a,img_real_sharp_a_fromOri[0]))
        
        # print(ssim(img_real_sharp_a,img_real_sharp_a))
        # print(ssim(img_real_sharp_a_fromOri[0],img_real_sharp_a_fromOri[0]))
        
        # print(ssim(img_fake_sharp_a_fromG,img_real_sharp_a_fromOri[0])) 
        # print(ssim(img_fake_sharp_a_fromG,img_real_sharp_a))

        ret['psnr_Gs_0S'] = psnr_Gs_0S
        ret['psnr_Rs_0S'] = psnr_Rs_0S
        ret['ssim_Gs_0S'] = ssim_Gs_0S
        ret['ssim_Rs_0S'] = ssim_Rs_0S
        ret['psnr_Gs_Ps'] = psnr_Gs_Ps
        ret['psnr_Rs_Ps'] = psnr_Rs_Ps
        ret['ssim_Gs_Ps'] = ssim_Gs_Ps
        ret['ssim_Rs_Ps'] = ssim_Rs_Ps
        return ret

    def forward_dummy(self, img):
        """Used for computing network FLOPs.

        Args:
            img (Tensor): Dummy input used to compute FLOPs.

        Returns:
            Tensor: Dummy output produced by forwarding the dummy input.
        """
        out = self.generator(img)
        return out

    def forward(self, img_sharp_real, img_blur_real, noise_map, ori_img_sharp_real, meta, test_mode=False, **kwargs):
        """Forward function.

        Args:
            img_a (Tensor): Input image from domain A.
            img_b (Tensor): Input image from domain B.
            meta (list[dict]): Input meta data.
            test_mode (bool): Whether in test mode or not. Default: False.
            kwargs (dict): Other arguments.
        """
        if test_mode:
            return self.forward_test(img_sharp_real, img_blur_real, noise_map, ori_img_sharp_real, meta= meta, **kwargs)

        return self.forward_train(img_sharp_real, img_blur_real,noise_map, meta)

    def backward_bgan_discriminator(self, outputs):
        """Backward function for the discriminator.

        Args:
            outputs (dict): Dict of forward results.

        Returns:
            dict: Loss dict.
        """
        # GAN loss for the discriminator
        losses = dict()

        fake_blur_logits = self.bgan_discriminator(outputs['fake_blur_a'].detach())
        real_blur_logits = self.bgan_discriminator(outputs['real_blur_b'].detach())
        if self.use_RBLoss:
            fake_blur_logis_rbl = fake_blur_logits - torch.mean(real_blur_logits)
            real_blur_logits_rbl = real_blur_logits - torch.mean(fake_blur_logits)
            losses['loss_gan_d_fake'] = self.bgan_loss(fake_blur_logis_rbl, target_is_real = False, is_disc = True)
            losses['loss_gan_d_real'] = self.bgan_loss(real_blur_logits_rbl, target_is_real = True, is_disc = True)
        else:
           losses['loss_gan_d_fake'] = self.bgan_loss(fake_blur_logits, target_is_real = False, is_disc = True)
           losses['loss_gan_d_real'] = self.bgan_loss(real_blur_logits, target_is_real = True, is_disc = True)

        loss_d, log_vars_d = self.parse_losses(losses)
        loss_d *= 0.5
        loss_d.backward(retain_graph=True)
        return log_vars_d


    def backward_dbgan_discriminator(self, outputs):
        """Backward function for the discriminator.

        Args:
            outputs (dict): Dict of forward results.

        Returns:
            dict: Loss dict.
        """
        # GAN loss for the discriminator
        losses = dict()

        fake_sharp_logits = self.dbgan_discriminator(outputs['fake_sharp_a'].detach())
        real_sharp_logits = self.dbgan_discriminator(outputs['real_sharp_a'].detach())
        
        if self.use_RBLoss:
            fake_sharp_logis_rbl = fake_sharp_logits - torch.mean(real_sharp_logits)
            real_sharp_logits_rbl = real_sharp_logits - torch.mean(fake_sharp_logits)
            losses['loss_dbgan_d_fake'] = self.dbgan_loss(fake_sharp_logis_rbl, target_is_real = False, is_disc = True)
            losses['loss_dbgan_d_real'] = self.dbgan_loss(real_sharp_logits_rbl, target_is_real = True, is_disc = True)
        else:
           losses['loss_dbgan_d_fake'] = self.dbgan_loss(fake_sharp_logits, target_is_real = False, is_disc = True)
           losses['loss_dbgan_d_real'] = self.dbgan_loss(real_sharp_logits, target_is_real = True, is_disc = True)

        loss_d, log_vars_d = self.parse_losses(losses)
        loss_d *= 0.5
        loss_d.backward(retain_graph=True)
        return log_vars_d


    def backward_bgan_generator(self, outputs):
        """Backward function for the generator.

        Args:
            outputs (dict): Dict of forward results.

        Returns:
            dict: Loss dict.
        """
        losses = dict()
        # GAN loss for the generator

        fake_blur_a_pred_logits = self.bgan_discriminator(outputs['fake_blur_a'])
        losses['loss_gan_g'] = self.bgan_loss(
            fake_blur_a_pred_logits, target_is_real=True, is_disc=False)
        # percputal loss for generator

        losses['logg_gan_g_perceptual'] = self.perceptual_loss(outputs['fake_blur_a'],outputs['real_sharp_a'])
        loss_g, log_vars_g = self.parse_losses(losses)
        loss_g.backward(retain_graph=True)
        return log_vars_g

    def backward_dbgan_generator(self, outputs):
        """Backward function for the generator.

        Args:
            outputs (dict): Dict of forward results.

        Returns:
            dict: Loss dict.
        """
        losses = dict()
        # GAN loss for the generator

        fake_sharp_a_pred_logits = self.dbgan_discriminator(outputs['fake_sharp_a'])
        losses['loss_dbgan_g'] = self.dbgan_loss(
            fake_sharp_a_pred_logits, target_is_real=True, is_disc=False)
        # percputal loss for generator
        losses['loss_dbgan_g_perceptual'] = self.perceptual_loss(outputs['fake_sharp_a'],outputs['fake_blur_a']) #//?
        # content loss for generator
        losses['loss_dbgan_g_content'] = self.content_loss(outputs['fake_sharp_a'], outputs['real_sharp_a'])
        loss_g, log_vars_g = self.parse_losses(losses)
        loss_g.backward(retain_graph=True)
        return log_vars_g


    def train_step(self, data_batch, optimizer):
        """Training step function.

        Args:
            data_batch (dict): Dict of the input data batch.
            optimizer (dict[torch.optim.Optimizer]): Dict of optimizers for
                the generator and discriminator.

        Returns:
            dict: Dict of loss, information for logger, the number of samples\
                and results for visualization.
        """
        # useful img data
        real_sharp_a, real_blur_b, noise_map, real_shape_a_ori  = data_batch['img_sharp_real'], data_batch['img_blur_real'], data_batch['noise_map'], data_batch['ori_img_sharp_real']
      
        meta = data_batch['meta']

        # forward generator
        outputs = self.forward(real_sharp_a, real_blur_b, noise_map,real_shape_a_ori,meta, test_mode=False)

        log_vars = dict()

        # discriminator
        set_requires_grad(self.bgan_discriminator, True)
        set_requires_grad(self.dbgan_discriminator,True)
        # optimize
        optimizer['bgan_discriminator'].zero_grad()
        optimizer['dbgan_discriminator'].zero_grad()

        log_vars.update(self.backward_bgan_discriminator(outputs=outputs))
        log_vars.update(self.backward_dbgan_discriminator(outputs=outputs))
        optimizer['bgan_discriminator'].step()
        optimizer['dbgan_discriminator'].step()

        # generator, no updates to discriminator parameters.
        if (self.step_counter % self.disc_steps == 0
                and self.step_counter >= self.disc_init_steps):
            set_requires_grad(self.bgan_discriminator, False)
            set_requires_grad(self.dbgan_discriminator, False)

            # optimize
            optimizer['bgan_generator'].zero_grad()
            optimizer['dbgan_generator'].zero_grad()
            log_vars.update(self.backward_bgan_generator(outputs=outputs))
            log_vars.update(self.backward_dbgan_generator(outputs=outputs))
            optimizer['bgan_generator'].step()
            optimizer['dbgan_generator'].step()

        self.step_counter += 1

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        results = dict(
            log_vars=log_vars,
            num_samples=len(outputs['real_sharp_a']),
            results=dict(
                fake_blur=outputs['fake_blur_a'].cpu(),
                fake_sharp=outputs['fake_sharp_a'].cpu(),
                real_sharp=outputs['real_sharp_a'].cpu()
            ))
        return results

    def val_step(self, data_batch, **kwargs):
        """Validation step function.

        Args:
            data_batch (dict): Dict of the input data batch.
            kwargs (dict): Other arguments.

        Returns:
            dict: Dict of evaluation results for validation.
        """
        # data
        real_sharp_a = data_batch['img_sharp_real']
        real_blur_b = data_batch['img_blur_real']
        noise_map = data_batch['noise_map']
        meta = data_batch['meta']
        real_sharp_a_ori = data_batch['ori_img_sharp_real']
        # forward generator
        results = self.forward(real_sharp_a, real_blur_b,noise_map, real_sharp_a_ori,meta,test_mode=1,**kwargs)

        # forward generator

        return results
