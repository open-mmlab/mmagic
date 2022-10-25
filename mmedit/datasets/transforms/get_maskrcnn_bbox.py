# Copyright (c) OpenMMLab. All rights reserved.
import cv2 as cv
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from mmcv.transforms import BaseTransform

from mmedit.registry import TRANSFORMS


@TRANSFORMS.register_module()
class InstanceCrop(BaseTransform):
    """## Arguments:

    - pred_data_path: Detectron2 predict results
    - box_num_upbound: object bounding boxes number.
                Default: -1 means use all the instances.
    """

    def __init__(self,
                 config_file,
                 key='img',
                 box_num_upbound=-1,
                 finesize=256):
        # detector
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(config_file))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
        self.predictor = DefaultPredictor(cfg)

        self.key = key
        self.box_num_upbound = box_num_upbound
        self.final_size = finesize

    def transform(self, results: dict) -> dict:

        # get consistent box prediction based on L channel
        full_img = results['img']
        # cv.imwrite('full_img.jpg', full_img)
        full_img_size = results['ori_img_shape'][:-1][::-1]
        lab_image = cv.cvtColor(full_img, cv.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv.split(lab_image)
        l_stack = np.stack([l_channel, l_channel, l_channel], axis=2)
        outputs = self.predictor(l_stack)

        # get the most confident boxes
        pred_bbox = outputs['instances'].pred_boxes.to(
            torch.device('cpu')).tensor.numpy()
        pred_scores = outputs['instances'].scores.cpu().data.numpy()
        pred_bbox = pred_bbox.astype(np.int32)
        if self.box_num_upbound > 0 and pred_bbox.shape[
                0] > self.box_num_upbound:
            index_mask = np.argsort(pred_scores, axis=0)
            index_mask = index_mask[pred_scores.shape[0] -
                                    self.box_num_upbound:pred_scores.shape[0]]
            pred_bbox = pred_bbox[index_mask]

        # get cropped images and box info
        cropped_img_list = []
        index_list = range(len(pred_bbox))
        box_info, box_info_2x, box_info_4x, box_info_8x = np.zeros(
            (4, len(index_list), 6))
        for i in index_list:
            startx, starty, endx, endy = pred_bbox[i]
            cropped_img = full_img[starty:endy, startx:endx, :]
            # cv.imwrite(f"crop_{i}.jpg", cropped_img)
            cropped_img_list.append(cropped_img)
            box_info[i] = np.array(
                get_box_info(pred_bbox[i], full_img_size, self.final_size))
            box_info_2x[i] = np.array(
                get_box_info(pred_bbox[i], full_img_size,
                             self.final_size // 2))
            box_info_4x[i] = np.array(
                get_box_info(pred_bbox[i], full_img_size,
                             self.final_size // 4))
            box_info_8x[i] = np.array(
                get_box_info(pred_bbox[i], full_img_size,
                             self.final_size // 8))

        # update results
        if len(pred_bbox) > 0:
            results['cropped_img'] = cropped_img_list
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
        return results


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
