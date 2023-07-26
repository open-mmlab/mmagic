# Copyright (c) OpenMMLab. All rights reserved.
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.data.transforms import paired_random_crop
from torchvision.transforms.functional_tensor import rgb_to_grayscale

# undo
from .basicsr_utils import DiffJPEG
from .diffusionmodules import Decoder_Mix, Encoder
from .distributions import DiagonalGaussianDistribution

# loss issue
# from .ldm import instantiate_from_config


def generate_poisson_noise_pt(img, scale=1.0, gray_noise=0):
    """Generate a batch of poisson noise (PyTorch version)

    Args:
        img (Tensor): Input image, shape (b, c, h, w), range [0, 1], float32.
        scale (float | Tensor): Noise scale. Number or Tensor with shape (b).
            Default: 1.0.
        gray_noise (float | Tensor): 0-1 number or Tensor with shape (b).
            0 for False, 1 for True. Default: 0.

    Returns:
        (Tensor): Returned noisy image, shape (b, c, h, w), range[0, 1],
            float32.
    """
    b, _, h, w = img.size()
    if isinstance(gray_noise, (float, int)):
        cal_gray_noise = gray_noise > 0
    else:
        gray_noise = gray_noise.view(b, 1, 1, 1)
        cal_gray_noise = torch.sum(gray_noise) > 0
    if cal_gray_noise:
        img_gray = rgb_to_grayscale(img, num_output_channels=1)
        # round and clip image for counting vals correctly
        img_gray = torch.clamp((img_gray * 255.0).round(), 0, 255) / 255.
        # use for-loop to get the unique values for each sample
        vals_list = [len(torch.unique(img_gray[i, :, :, :])) for i in range(b)]
        vals_list = [2**np.ceil(np.log2(vals)) for vals in vals_list]
        vals = img_gray.new_tensor(vals_list).view(b, 1, 1, 1)
        out = torch.poisson(img_gray * vals) / vals
        noise_gray = out - img_gray
        noise_gray = noise_gray.expand(b, 3, h, w)

    # always calculate color noise
    # round and clip image for counting vals correctly
    img = torch.clamp((img * 255.0).round(), 0, 255) / 255.
    # use for-loop to get the unique values for each sample
    vals_list = [len(torch.unique(img[i, :, :, :])) for i in range(b)]
    vals_list = [2**np.ceil(np.log2(vals)) for vals in vals_list]
    vals = img.new_tensor(vals_list).view(b, 1, 1, 1)
    out = torch.poisson(img * vals) / vals
    noise = out - img
    if cal_gray_noise:
        noise = noise * (1 - gray_noise) + noise_gray * gray_noise
    if not isinstance(scale, (float, int)):
        scale = scale.view(b, 1, 1, 1)
    return noise * scale


def generate_gaussian_noise_pt(img, sigma=10, gray_noise=0):
    """Add Gaussian noise (PyTorch version).

    Args:
        img (Tensor): Shape (b, c, h, w), range[0, 1], float32.
        scale (float | Tensor): Noise scale. Default: 1.0.

    Returns:
        (Tensor): Returned noisy image, shape (b, c, h, w), range[0, 1],
            float32.
    """
    b, _, h, w = img.size()
    if not isinstance(sigma, (float, int)):
        sigma = sigma.view(img.size(0), 1, 1, 1)
    if isinstance(gray_noise, (float, int)):
        cal_gray_noise = gray_noise > 0
    else:
        gray_noise = gray_noise.view(b, 1, 1, 1)
        cal_gray_noise = torch.sum(gray_noise) > 0

    if cal_gray_noise:
        noise_gray = torch.randn(
            *img.size()[2:4], dtype=img.dtype,
            device=img.device) * sigma / 255.
        noise_gray = noise_gray.view(b, 1, h, w)

    # always calculate color noise
    noise = torch.randn(
        *img.size(), dtype=img.dtype, device=img.device) * sigma / 255.

    if cal_gray_noise:
        noise = noise * (1 - gray_noise) + noise_gray * gray_noise
    return noise


def random_generate_gaussian_noise_pt(img, sigma_range=(0, 10), gray_prob=0):
    sigma = torch.rand(
        img.size(0), dtype=img.dtype,
        device=img.device) * (sigma_range[1] - sigma_range[0]) + sigma_range[0]
    gray_noise = torch.rand(img.size(0), dtype=img.dtype, device=img.device)
    gray_noise = (gray_noise < gray_prob).float()
    return generate_gaussian_noise_pt(img, sigma, gray_noise)


def random_add_gaussian_noise_pt(img,
                                 sigma_range=(0, 1.0),
                                 gray_prob=0,
                                 clip=True,
                                 rounds=False):
    noise = random_generate_gaussian_noise_pt(img, sigma_range, gray_prob)
    out = img + noise
    if clip and rounds:
        out = torch.clamp((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = torch.clamp(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.
    return out


def random_generate_poisson_noise_pt(img, scale_range=(0, 1.0), gray_prob=0):
    scale = torch.rand(
        img.size(0), dtype=img.dtype,
        device=img.device) * (scale_range[1] - scale_range[0]) + scale_range[0]
    gray_noise = torch.rand(img.size(0), dtype=img.dtype, device=img.device)
    gray_noise = (gray_noise < gray_prob).float()
    return generate_poisson_noise_pt(img, scale, gray_noise)


def random_add_poisson_noise_pt(img,
                                scale_range=(0, 1.0),
                                gray_prob=0,
                                clip=True,
                                rounds=False):
    noise = random_generate_poisson_noise_pt(img, scale_range, gray_prob)
    out = img + noise
    if clip and rounds:
        out = torch.clamp((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = torch.clamp(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.
    return out


def filter2D(img, kernel):
    """PyTorch version of cv2.filter2D.

    Args:
        img (Tensor): (b, c, h, w)
        kernel (Tensor): (b, k, k)
    """
    k = kernel.size(-1)
    b, c, h, w = img.size()
    if k % 2 == 1:
        img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode='reflect')
    else:
        raise ValueError('Wrong kernel size')

    ph, pw = img.size()[-2:]

    if kernel.size(0) == 1:
        # apply the same kernel to all batch images
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1,
                                                1).view(b * c, 1, k, k)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)


class USMSharp(nn.Module):

    def __init__(self, radius=50, sigma=0):
        super(USMSharp, self).__init__()
        if radius % 2 == 0:
            radius += 1
        self.radius = radius
        kernel = cv2.getGaussianKernel(radius, sigma)
        kernel = torch.FloatTensor(np.dot(kernel,
                                          kernel.transpose())).unsqueeze_(0)
        self.register_buffer('kernel', kernel)

    def forward(self, img, weight=0.5, threshold=10):
        blur = filter2D(img, self.kernel)
        residual = img - blur

        mask = torch.abs(residual) * 255 > threshold
        mask = mask.float()
        soft_mask = filter2D(mask, self.kernel)
        sharp = img + weight * residual
        sharp = torch.clip(sharp, 0, 1)
        return soft_mask * sharp + (1 - soft_mask) * img


# TODO REFACTOR
class AutoencoderKLResi(nn.Module):

    def __init__(
        self,
        ddconfig,
        lossconfig,
        embed_dim,
        ckpt_path=None,
        ignore_keys=[],
        image_key='image',
        colorize_nlabels=None,
        monitor=None,
        fusion_w=1.0,
        freeze_dec=True,
        synthesis_data=False,
        use_usm=False,
        test_gt=False,
    ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder_Mix(**ddconfig)
        self.decoder.fusion_w = fusion_w
        # self.loss = instantiate_from_config(lossconfig) # build
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig['z_channels'],
                                          2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim,
                                               ddconfig['z_channels'], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer('colorize',
                                 torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            missing_list = self.init_from_ckpt(
                ckpt_path, ignore_keys=ignore_keys)
        else:
            missing_list = []

        print('>>>>>>>>>>>>>>>>>missing>>>>>>>>>>>>>>>>>>>')
        print(missing_list)
        self.synthesis_data = synthesis_data
        self.use_usm = use_usm
        self.test_gt = test_gt

        if freeze_dec:
            for name, param in self.named_parameters():
                if 'fusion_layer' in name:
                    param.requires_grad = True
                # elif 'encoder' in name:
                #     param.requires_grad = True
                # elif 'quant_conv' in name and 'post_quant_conv' not in name:
                #     param.requires_grad = True
                elif 'loss.discriminator' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        print('>>>>>>>>>>>>>>>>>trainable_list>>>>>>>>>>>>>>>>>>>')
        trainable_list = []
        for name, params in self.named_parameters():
            if params.requires_grad:
                trainable_list.append(name)
        print(trainable_list)

        print('>>>>>>>>>>>>>>>>>Untrainable_list>>>>>>>>>>>>>>>>>>>')
        untrainable_list = []
        for name, params in self.named_parameters():
            if not params.requires_grad:
                untrainable_list.append(name)
        print(untrainable_list)
        # print('>>>>>>>>>>>>>>>>>untrainable_list>>>>>>>>>>>>>>>>>>>')
        # print(untrainable_list)

    # def init_from_ckpt(self, path, ignore_keys=list()):
    #     sd = torch.load(path, map_location="cpu")["state_dict"]
    #     keys = list(sd.keys())
    #     for k in keys:
    #         for ik in ignore_keys:
    #             if k.startswith(ik):
    #                 print("Deleting key {} from state_dict.".format(k))
    #                 del sd[k]
    #     self.load_state_dict(sd, strict=False)
    #     print(f"Restored from {path}")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location='cpu')
        if 'state_dict' in list(sd.keys()):
            sd = sd['state_dict']
        keys = list(sd.keys())
        for k in keys:
            if 'first_stage_model' in k:
                sd[k[18:]] = sd[k]
                del sd[k]
            for ik in ignore_keys:
                if k.startswith(ik):
                    print('Deleting key {} from state_dict.'.format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(
            sd,
            strict=False) if not only_model else self.model.load_state_dict(
                sd, strict=False)
        print(f'Encoder Restored from {path} with {len(missing)} '
              'missing and {len(unexpected)} unexpected keys')
        if len(missing) > 0:
            print(f'Missing Keys: {missing}')
        if len(unexpected) > 0:
            print(f'Unexpected Keys: {unexpected}')
        return missing

    def encode(self, x):
        h, enc_fea = self.encoder(x, return_fea=True)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        # posterior = h
        return posterior, enc_fea

    def encode_gt(self, x, new_encoder):
        h = new_encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior, moments

    def decode(self, z, enc_fea):
        z = self.post_quant_conv(z)
        dec = self.decoder(z, enc_fea)
        return dec

    def forward(self, input, latent, sample_posterior=True):
        posterior, enc_fea_lq = self.encode(input)
        dec = self.decode(latent, enc_fea_lq)
        return dec, posterior

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a
        batch.

        Batch processing limits the diversity of synthetic degradations in a
        batch. For example, samples in a batch could not have different resize
        scaling factors. Therefore, we employ this training pair pool to
        increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        _, c_, h_, w_ = self.latent.size()
        if b == self.configs.data.params.batch_size:
            if not hasattr(self, 'queue_size'):
                self.queue_size = self.configs.data.params.train.params.get(
                    'queue_size', b * 50)
            if not hasattr(self, 'queue_lr'):
                assert self.queue_size % b == 0, \
                    f'queue size {self.queue_size} should be' \
                    'divisible by batch size {b}'
                self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
                _, c, h, w = self.gt.size()
                self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
                self.queue_sample = torch.zeros(self.queue_size, c, h,
                                                w).cuda()
                self.queue_latent = torch.zeros(self.queue_size, c_, h_,
                                                w_).cuda()
                self.queue_ptr = 0
            if self.queue_ptr == self.queue_size:  # the pool is full
                # do dequeue and enqueue
                # shuffle
                idx = torch.randperm(self.queue_size)
                self.queue_lr = self.queue_lr[idx]
                self.queue_gt = self.queue_gt[idx]
                self.queue_sample = self.queue_sample[idx]
                self.queue_latent = self.queue_latent[idx]
                # get first b samples
                lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
                gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
                sample_dequeue = self.queue_sample[0:b, :, :, :].clone()
                latent_dequeue = self.queue_latent[0:b, :, :, :].clone()
                # update the queue
                self.queue_lr[0:b, :, :, :] = self.lq.clone()
                self.queue_gt[0:b, :, :, :] = self.gt.clone()
                self.queue_sample[0:b, :, :, :] = self.sample.clone()
                self.queue_latent[0:b, :, :, :] = self.latent.clone()

                self.lq = lq_dequeue
                self.gt = gt_dequeue
                self.sample = sample_dequeue
                self.latent = latent_dequeue
            else:
                # only do enqueue
                self.queue_lr[self.queue_ptr:self.queue_ptr +
                              b, :, :, :] = self.lq.clone()
                self.queue_gt[self.queue_ptr:self.queue_ptr +
                              b, :, :, :] = self.gt.clone()
                self.queue_sample[self.queue_ptr:self.queue_ptr +
                                  b, :, :, :] = self.sample.clone()
                self.queue_latent[self.queue_ptr:self.queue_ptr +
                                  b, :, :, :] = self.latent.clone()
                self.queue_ptr = self.queue_ptr + b

    def get_input(self, batch):
        input = batch['lq']
        gt = batch['gt']
        latent = batch['latent']
        sample = batch['sample']

        assert not torch.isnan(latent).any()

        input = input.to(memory_format=torch.contiguous_format).float()
        gt = gt.to(memory_format=torch.contiguous_format).float()
        latent = latent.to(
            memory_format=torch.contiguous_format).float() / 0.18215

        gt = gt * 2.0 - 1.0
        input = input * 2.0 - 1.0
        sample = sample * 2.0 - 1.0

        return input, gt, latent, sample

    @torch.no_grad()
    def get_input_synthesis(self, batch, val=False, test_gt=False):

        jpeger = DiffJPEG(
            differentiable=False).cuda()  # simulate JPEG compression artifacts
        im_gt = batch['gt'].cuda()
        if self.use_usm:
            usm_sharpener = USMSharp().cuda()  # do usm sharpening
            im_gt = usm_sharpener(im_gt)
        im_gt = im_gt.to(memory_format=torch.contiguous_format).float()
        kernel1 = batch['kernel1'].cuda()
        kernel2 = batch['kernel2'].cuda()
        sinc_kernel = batch['sinc_kernel'].cuda()

        ori_h, ori_w = im_gt.size()[2:4]

        # The first degradation process ----------------------- #
        # blur
        out = filter2D(im_gt, kernel1)
        # random resize
        updown_type = random.choices(
            ['up', 'down', 'keep'],
            self.configs.degradation['resize_prob'],
        )[0]
        if updown_type == 'up':
            scale = random.uniform(1,
                                   self.configs.degradation['resize_range'][1])
        elif updown_type == 'down':
            scale = random.uniform(self.configs.degradation['resize_range'][0],
                                   1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        # add noise
        gray_noise_prob = self.configs.degradation['gray_noise_prob']
        if random.random() < self.configs.degradation['gaussian_noise_prob']:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=self.configs.degradation['noise_range'],
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob,
            )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.configs.degradation['poisson_scale_range'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(
            *self.configs.degradation['jpeg_range'])
        out = torch.clamp(out, 0, 1)
        # clamp to [0, 1], otherwise JPEGer result in unpleasant artifacts
        out = jpeger(out, quality=jpeg_p)
        # The second degradation process ----------------------- #
        # blur
        if random.random() < self.configs.degradation['second_blur_prob']:
            out = filter2D(out, kernel2)
        # random resize
        updown_type = random.choices(
            ['up', 'down', 'keep'],
            self.configs.degradation['resize_prob2'],
        )[0]
        if updown_type == 'up':
            scale = random.uniform(
                1, self.configs.degradation['resize_range2'][1])
        elif updown_type == 'down':
            scale = random.uniform(
                self.configs.degradation['resize_range2'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
            out,
            size=(int(ori_h / self.configs.sf * scale),
                  int(ori_w / self.configs.sf * scale)),
            mode=mode,
        )
        # add noise
        gray_noise_prob = self.configs.degradation['gray_noise_prob2']
        if random.random() < self.configs.degradation['gaussian_noise_prob2']:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=self.configs.degradation['noise_range2'],
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob,
            )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.configs.degradation['poisson_scale_range2'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False,
            )

        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes.
        # We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically,
        # we find other combinations (sinc + JPEG + Resize)
        # will introduce twisted lines.
        if random.random() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                out,
                size=(ori_h // self.configs.sf, ori_w // self.configs.sf),
                mode=mode,
            )
            out = filter2D(out, sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(
                *self.configs.degradation['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(
                *self.configs.degradation['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                out,
                size=(ori_h // self.configs.sf, ori_w // self.configs.sf),
                mode=mode,
            )
            out = filter2D(out, sinc_kernel)

        # clamp and round
        im_lq = torch.clamp(out, 0, 1.0)

        # random crop
        gt_size = self.configs.degradation['gt_size']
        im_gt, im_lq = paired_random_crop(im_gt, im_lq, gt_size,
                                          self.configs.sf)
        self.lq, self.gt = im_lq, im_gt

        self.lq = F.interpolate(
            self.lq,
            size=(self.gt.size(-2), self.gt.size(-1)),
            mode='bicubic',
        )

        self.latent = batch['latent'] / 0.18215
        self.sample = batch['sample'] * 2 - 1.0
        # training pair pool
        if not val:
            self._dequeue_and_enqueue()
        # sharpen self.gt again,
        # as we have changed the self.gt with self._dequeue_and_enqueue
        self.lq = self.lq.contiguous()
        # for the warning:
        # grad and param do not obey the gradient layout contract
        self.lq = self.lq * 2 - 1.0
        self.gt = self.gt * 2 - 1.0

        self.lq = torch.clamp(self.lq, -1.0, 1.0)

        x = self.lq
        y = self.gt
        x = x.to(self.device)
        y = y.to(self.device)

        if self.test_gt:
            return y, y, self.latent.to(self.device), self.sample.to(
                self.device)
        else:
            return x, y, self.latent.to(self.device), self.sample.to(
                self.device)

    def training_step(self, batch, batch_idx, optimizer_idx):
        if self.synthesis_data:
            inputs, gts, latents, _ = self.get_input_synthesis(
                batch, val=False)
        else:
            inputs, gts, latents, _ = self.get_input(batch)
        reconstructions, posterior = self(inputs, latents)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(
                gts,
                reconstructions,
                posterior,
                optimizer_idx,
                self.global_step,
                last_layer=self.get_last_layer(),
                split='train')
            self.log(
                'aeloss',
                aeloss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True)
            self.log_dict(
                log_dict_ae,
                prog_bar=False,
                logger=True,
                on_step=True,
                on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(
                gts,
                reconstructions,
                posterior,
                optimizer_idx,
                self.global_step,
                last_layer=self.get_last_layer(),
                split='train')

            self.log(
                'discloss',
                discloss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True)
            self.log_dict(
                log_dict_disc,
                prog_bar=False,
                logger=True,
                on_step=True,
                on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs, gts, latents, _ = self.get_input(batch)

        reconstructions, posterior = self(inputs, latents)
        aeloss, log_dict_ae = self.loss(
            gts,
            reconstructions,
            posterior,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split='val')

        discloss, log_dict_disc = self.loss(
            gts,
            reconstructions,
            posterior,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split='val')

        self.log('val/rec_loss', log_dict_ae['val/rec_loss'])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()) +
            # list(self.quant_conv.parameters())+
            list(self.post_quant_conv.parameters()),
            lr=lr,
            betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        if self.synthesis_data:
            x, gts, latents, samples = self.get_input_synthesis(
                batch, val=False)
        else:
            x, gts, latents, samples = self.get_input(batch)
        x = x.to(self.device)
        latents = latents.to(self.device)
        samples = samples.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x, latents)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                gts = self.to_rgb(gts)
                samples = self.to_rgb(samples)
                xrec = self.to_rgb(xrec)
            # log["samples"] =
            # self.decode(torch.randn_like(posterior.sample()))
            log['reconstructions'] = xrec
        log['inputs'] = x
        log['gts'] = gts
        log['samples'] = samples
        return log

    def to_rgb(self, x):
        assert self.image_key == 'segmentation'
        if not hasattr(self, 'colorize'):
            self.register_buffer('colorize',
                                 torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x
