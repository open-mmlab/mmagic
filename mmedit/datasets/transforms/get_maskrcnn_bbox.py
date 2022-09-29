# Copyright (c) OpenMMLab. All rights reserved.
from random import sample

import cv2 as cv
import numpy as np
import torch
import torchvision.transforms as transforms
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from PIL import Image
from skimage import color

from mmedit.registry import TRANSFORMS


@TRANSFORMS.register_module()
class GenMaskRCNNBbox:

    def __init__(self, key='gt', stage='test_fusion', finesize=256):
        self.key = key
        self.predictor = self.detectron()
        self.stage = stage
        self.final_size = finesize
        self.transforms = transforms.Compose([
            transforms.Resize((self.final_size, self.final_size),
                              interpolation=2),
            transforms.ToTensor()
        ])

    def gen_maskrcnn_bbox_fromPred(self,
                                   img,
                                   bbox_path=None,
                                   box_num_upbound=8):
        '''
        ## Arguments:
        - pred_data_path: Detectron2 predict results
        - box_num_upbound: object bounding boxes number.
                           Default: -1 means use all the instances.
        '''
        if bbox_path:
            pred_data = np.load(bbox_path)
            pred_bbox = pred_data['bbox'].astype(np.int32)
            pred_scores = pred_data['scores']
        else:
            lab_image = cv.cvtColor(img, cv.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv.split(lab_image)
            l_stack = np.stack([l_channel, l_channel, l_channel], axis=2)
            outputs = self.predictor(l_stack)
            pred_bbox = outputs['instances'].pred_boxes.to(
                torch.device('cpu')).tensor.numpy()
            pred_scores = outputs['instances'].scores.cpu().data.numpy()

            pred_bbox = pred_bbox.astype(np.int32)
        if 0 < box_num_upbound < pred_bbox.shape[0]:
            index_mask = np.argsort(
                pred_scores, axis=0)[pred_scores.shape[0] -
                                     box_num_upbound:pred_scores.shape[0]]
            pred_bbox = pred_bbox[index_mask]

        return pred_bbox

    @staticmethod
    def gen_gray_color_pil(rgb_img):
        '''
        return: RGB and GRAY pillow image object
        '''
        if len(np.asarray(rgb_img).shape) == 2:
            rgb_img = np.stack([
                np.asarray(rgb_img),
                np.asarray(rgb_img),
                np.asarray(rgb_img)
            ], 2)
            rgb_img = Image.fromarray(rgb_img)
        gray_img = np.round(color.rgb2gray(np.asarray(rgb_img)) *
                            255.0).astype(np.uint8)
        gray_img = np.stack([gray_img, gray_img, gray_img], -1)
        gray_img = Image.fromarray(gray_img)
        return rgb_img, gray_img

    @staticmethod
    def read_to_pil(out_img):
        '''
        return: pillow image object HxWx3
        '''
        out_img = Image.fromarray(out_img)
        if len(np.asarray(out_img).shape) == 2:
            out_img = np.stack([
                np.asarray(out_img),
                np.asarray(out_img),
                np.asarray(out_img)
            ], 2)
            out_img = Image.fromarray(out_img)
        return out_img

    @staticmethod
    def get_box_info(pred_bbox, original_shape, final_size):
        assert len(pred_bbox) == 4
        resize_startx = int(pred_bbox[0] / original_shape[0] * final_size)
        resize_starty = int(pred_bbox[1] / original_shape[1] * final_size)
        resize_endx = int(pred_bbox[2] / original_shape[0] * final_size)
        resize_endy = int(pred_bbox[3] / original_shape[1] * final_size)
        rh = resize_endx - resize_startx
        rw = resize_endy - resize_starty
        if rh < 1:
            if final_size - resize_endx > 1:
                resize_endx += 1
            else:
                resize_startx -= 1
            rh = 1
        if rw < 1:
            if final_size - resize_endy > 1:
                resize_endy += 1
            else:
                resize_starty -= 1
            rw = 1
        L_pad = resize_startx
        R_pad = final_size - resize_endx
        T_pad = resize_starty
        B_pad = final_size - resize_endy
        return [L_pad, R_pad, T_pad, B_pad, rh, rw]

    def test_fusion(self, results):
        img = results['gt']
        pil_img = self.read_to_pil(img)
        if results['bbox_path']:
            pred_bbox = self.gen_maskrcnn_bbox_fromPred(
                img, results['bbox_path'], box_num_upbound=8)
        else:
            pred_bbox = self.gen_maskrcnn_bbox_fromPred(img, box_num_upbound=8)

        img_list = [self.transforms(pil_img)]  # 这里删除了一个transform

        cropped_img_list = []
        index_list = range(len(pred_bbox))
        box_info, box_info_2x, box_info_4x, box_info_8x = np.zeros(
            (4, len(index_list), 6))
        for i in index_list:
            startx, starty, endx, endy = pred_bbox[i]
            box_info[i] = np.array(
                self.get_box_info(pred_bbox[i], pil_img.size, self.final_size))
            box_info_2x[i] = np.array(
                self.get_box_info(pred_bbox[i], pil_img.size,
                                  self.final_size // 2))
            box_info_4x[i] = np.array(
                self.get_box_info(pred_bbox[i], pil_img.size,
                                  self.final_size // 4))
            box_info_8x[i] = np.array(
                self.get_box_info(pred_bbox[i], pil_img.size,
                                  self.final_size // 8))
            cropped_img = self.transforms(
                pil_img.crop((startx, starty, endx, endy)))
            cropped_img_list.append(cropped_img)

        results['full_img'] = torch.stack(img_list)
        # output['file_id'] = self.IMAGE_ID_LIST[index].split('.')[0]
        if len(pred_bbox) > 0:
            results['cropped_img'] = torch.stack(cropped_img_list)
            results['box_info'] = torch.from_numpy(box_info).type(torch.long)
            results['box_info_2x'] = torch.from_numpy(box_info_2x).type(
                torch.long)
            results['box_info_4x'] = torch.from_numpy(box_info_4x).type(
                torch.long)
            results['box_info_8x'] = torch.from_numpy(box_info_8x).type(
                torch.long)
            results['empty_box'] = False
        else:
            results['empty_box'] = True
        print('full_img:', results['full_img'].size)
        # print("cropped_img:", results['cropped_img'].size)
        return results

    def train(self, results):
        img = results[self.key]
        if results['bbox_path']:
            pred_bbox = self.gen_maskrcnn_bbox_fromPred(
                img, results['bbox_path'])
        else:
            pred_bbox = self.gen_maskrcnn_bbox_fromPred(img)
        rgb_img, gray_img = self.gen_gray_color_pil(img)
        index_list = range(len(pred_bbox))
        index_list = sample(index_list, 1)
        startx, starty, endx, endy = pred_bbox[index_list[0]]

        results['rgb_img'] = self.transforms(
            rgb_img.crop((startx, starty, endx, endy)))
        results['gray_img'] = self.transforms(
            gray_img.crop((startx, starty, endx, endy)))

        return results

    def __call__(self, results):

        if self.stage == 'test_fusion':
            results = self.test_fusion(results)

        if self.stage == 'train':
            results = self.train(results)

        return results

    def detectron(self):
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file(
                'COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml'))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        cfg.MODEL.WEIGHTS = '/mnt/d/code/MMEditing/model_final_2d9806.pkl'
        predictor = DefaultPredictor(cfg)
        return predictor
