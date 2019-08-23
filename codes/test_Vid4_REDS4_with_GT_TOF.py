"""
TOF testing script, test Vid4 (SR) and REDS4 (SR-clean) datasets
write to txt log file
"""

import os
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import torch

import utils.util as util
import data.util as data_util
import models.archs.TOF_arch as TOF_arch


def main():
    #################
    # configurations
    #################
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    data_mode = 'Vid4'  # Vid4 | sharp_bicubic (REDS)

    # model
    N_in = 7
    model_path = '../experiments/pretrained_models/TOF_official.pth'
    adapt_official = True if 'official' in model_path else False
    model = TOF_arch.TOFlow(adapt_official=adapt_official)

    #### dataset
    if data_mode == 'Vid4':
        test_dataset_folder = '../datasets/Vid4/BIx4up_direct/*'
    else:
        test_dataset_folder = '../datasets/REDS4/{}/*'.format(data_mode)

    #### evaluation
    crop_border = 0
    border_frame = N_in // 2  # border frames when evaluate
    # temporal padding mode
    padding = 'new_info'  # different from the official setting
    save_imgs = True
    ############################################################################
    device = torch.device('cuda')
    save_folder = '../results/{}'.format(data_mode)
    util.mkdirs(save_folder)
    util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')

    #### log info
    logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Save images: {}'.format(save_imgs))

    def read_image(img_path):
        '''read one image from img_path
        Return img: HWC, BGR, [0,1], numpy
        '''
        img_GT = cv2.imread(img_path)
        img = img_GT.astype(np.float32) / 255.
        return img

    def read_seq_imgs(img_seq_path):
        '''read a sequence of images'''
        img_path_l = sorted(glob.glob(img_seq_path + '/*'))
        img_l = [read_image(v) for v in img_path_l]
        # stack to TCHW, RGB, [0,1], torch
        imgs = np.stack(img_l, axis=0)
        imgs = imgs[:, :, :, [2, 1, 0]]
        imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2)))).float()
        return imgs

    def index_generation(crt_i, max_n, N, padding='reflection'):
        '''
        padding: replicate | reflection | new_info | circle
        '''
        max_n = max_n - 1
        n_pad = N // 2
        return_l = []

        for i in range(crt_i - n_pad, crt_i + n_pad + 1):
            if i < 0:
                if padding == 'replicate':
                    add_idx = 0
                elif padding == 'reflection':
                    add_idx = -i
                elif padding == 'new_info':
                    add_idx = (crt_i + n_pad) + (-i)
                elif padding == 'circle':
                    add_idx = N + i
                else:
                    raise ValueError('Wrong padding mode')
            elif i > max_n:
                if padding == 'replicate':
                    add_idx = max_n
                elif padding == 'reflection':
                    add_idx = max_n * 2 - i
                elif padding == 'new_info':
                    add_idx = (crt_i - n_pad) - (i - max_n)
                elif padding == 'circle':
                    add_idx = i - N
                else:
                    raise ValueError('Wrong padding mode')
            else:
                add_idx = i
            return_l.append(add_idx)
        return return_l

    def single_forward(model, imgs_in):
        with torch.no_grad():
            model_output = model(imgs_in)
            if isinstance(model_output, list) or isinstance(model_output, tuple):
                output = model_output[0]
            else:
                output = model_output
        return output

    sub_folder_l = sorted(glob.glob(test_dataset_folder))
    #### set up the models
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    avg_psnr_l, avg_psnr_center_l, avg_psnr_border_l = [], [], []
    sub_folder_name_l = []

    # for each sub-folder
    for sub_folder in sub_folder_l:
        sub_folder_name = sub_folder.split('/')[-1]
        sub_folder_name_l.append(sub_folder_name)
        save_sub_folder = osp.join(save_folder, sub_folder_name)

        img_path_l = sorted(glob.glob(sub_folder + '/*'))
        max_idx = len(img_path_l)

        if save_imgs:
            util.mkdirs(save_sub_folder)

        #### read LR images
        imgs = read_seq_imgs(sub_folder)
        #### read GT images
        img_GT_l = []
        if data_mode == 'Vid4':
            sub_folder_GT = osp.join(sub_folder.replace('/BIx4up_direct/', '/GT/'), '*')
        else:
            sub_folder_GT = osp.join(sub_folder.replace('/{}/'.format(data_mode), '/GT/'), '*')
        for img_GT_path in sorted(glob.glob(sub_folder_GT)):
            img_GT_l.append(read_image(img_GT_path))

        avg_psnr, avg_psnr_border, avg_psnr_center = 0, 0, 0
        cal_n_border, cal_n_center = 0, 0

        # process each image
        for img_idx, img_path in enumerate(img_path_l):
            c_idx = int(osp.splitext(osp.basename(img_path))[0])
            select_idx = index_generation(c_idx, max_idx, N_in, padding=padding)
            # get input images
            imgs_in = imgs.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0).to(device)
            output = single_forward(model, imgs_in)
            output_f = output.data.float().cpu().squeeze(0)

            output = util.tensor2img(output_f)

            # save imgs
            if save_imgs:
                cv2.imwrite(osp.join(save_sub_folder, '{:08d}.png'.format(c_idx)), output)

            #### calculate PSNR
            output = output / 255.
            GT = np.copy(img_GT_l[img_idx])
            # For REDS, evaluate on RGB channels; for Vid4, evaluate on Y channels
            if data_mode == 'Vid4':  # bgr2y, [0, 1]
                GT = data_util.bgr2ycbcr(GT)
                output = data_util.bgr2ycbcr(output)
            if crop_border == 0:
                cropped_output = output
                cropped_GT = GT
            else:
                cropped_output = output[crop_border:-crop_border, crop_border:-crop_border]
                cropped_GT = GT[crop_border:-crop_border, crop_border:-crop_border]
            crt_psnr = util.calculate_psnr(cropped_output * 255, cropped_GT * 255)
            logger.info('{:3d} - {:25}.png \tPSNR: {:.6f} dB'.format(img_idx + 1, c_idx, crt_psnr))

            if img_idx >= border_frame and img_idx < max_idx - border_frame:  # center frames
                avg_psnr_center += crt_psnr
                cal_n_center += 1
            else:  # border frames
                avg_psnr_border += crt_psnr
                cal_n_border += 1

        avg_psnr = (avg_psnr_center + avg_psnr_border) / (cal_n_center + cal_n_border)
        avg_psnr_center = avg_psnr_center / cal_n_center
        if cal_n_border == 0:
            avg_psnr_border = 0
        else:
            avg_psnr_border = avg_psnr_border / cal_n_border

        logger.info('Folder {} - Average PSNR: {:.6f} dB for {} frames; '
                    'Center PSNR: {:.6f} dB for {} frames; '
                    'Border PSNR: {:.6f} dB for {} frames.'.format(sub_folder_name, avg_psnr,
                                                                   (cal_n_center + cal_n_border),
                                                                   avg_psnr_center, cal_n_center,
                                                                   avg_psnr_border, cal_n_border))

        avg_psnr_l.append(avg_psnr)
        avg_psnr_center_l.append(avg_psnr_center)
        avg_psnr_border_l.append(avg_psnr_border)

    logger.info('################ Tidy Outputs ################')
    for name, psnr, psnr_center, psnr_border in zip(sub_folder_name_l, avg_psnr_l,
                                                    avg_psnr_center_l, avg_psnr_border_l):
        logger.info('Folder {} - Average PSNR: {:.6f} dB. '
                    'Center PSNR: {:.6f} dB. '
                    'Border PSNR: {:.6f} dB.'.format(name, psnr, psnr_center, psnr_border))
    logger.info('################ Final Results ################')
    logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Total Average PSNR: {:.6f} dB for {} clips. '
                'Center PSNR: {:.6f} dB. Border PSNR: {:.6f} dB.'.format(
                    sum(avg_psnr_l) / len(avg_psnr_l), len(sub_folder_l),
                    sum(avg_psnr_center_l) / len(avg_psnr_center_l),
                    sum(avg_psnr_border_l) / len(avg_psnr_border_l)))


if __name__ == '__main__':
    main()
