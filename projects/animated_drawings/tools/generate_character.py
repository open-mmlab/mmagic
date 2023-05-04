import os
from argparse import ArgumentParser

import cv2
import numpy as np
import PIL.Image as Image
import torch
import yaml
from controlnet_aux import OpenposeDetector
from controlnet_aux.open_pose.face import faceDetect
from controlnet_aux.open_pose.hand import handDetect
from controlnet_aux.open_pose.util import (HWC3, draw_bodypose, draw_facepose,
                                           draw_handpose, resize_image)


def draw_pose(pose, H, W, draw_body=True, draw_hand=True, draw_face=True):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    if draw_body:
        canvas = draw_bodypose(canvas, candidate, subset)

    if draw_hand:
        canvas = draw_handpose(canvas, hands)

    if draw_face:
        canvas = draw_facepose(canvas, faces)

    return canvas


class OpenposeDetectorPoint(OpenposeDetector):

    def __call__(self,
                 input_image,
                 detect_resolution=512,
                 image_resolution=512,
                 hand_and_face=False,
                 return_pil=True):
        # hand = False
        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)
        input_image = input_image[:, :, ::-1].copy()
        H, W, C = input_image.shape
        with torch.no_grad():
            candidate, subset = self.body_estimation(input_image)
            hands = []
            faces = []
            if hand_and_face:
                # Hand
                hands_list = handDetect(candidate, subset, input_image)
                for x, y, w, is_left in hands_list:
                    peaks = self.hand_estimation(
                        input_image[y:y + w, x:x + w, :]).astype(np.float32)
                    if peaks.ndim == 2 and peaks.shape[1] == 2:
                        peaks[:, 0] = np.where(peaks[:, 0] < 1e-6, -1,
                                               peaks[:, 0] + x) / float(W)
                        peaks[:, 1] = np.where(peaks[:, 1] < 1e-6, -1,
                                               peaks[:, 1] + y) / float(H)
                        hands.append(peaks.tolist())
                # Face
                faces_list = faceDetect(candidate, subset, input_image)
                for x, y, w in faces_list:
                    heatmaps = self.face_estimation(input_image[y:y + w,
                                                                x:x + w, :])
                    peaks = self.face_estimation.compute_peaks_from_heatmaps(
                        heatmaps).astype(np.float32)
                    if peaks.ndim == 2 and peaks.shape[1] == 2:
                        peaks[:, 0] = np.where(peaks[:, 0] < 1e-6, -1,
                                               peaks[:, 0] + x) / float(W)
                        peaks[:, 1] = np.where(peaks[:, 1] < 1e-6, -1,
                                               peaks[:, 1] + y) / float(H)
                        faces.append(peaks.tolist())

            if candidate.ndim == 2 and candidate.shape[1] == 4:
                candidate = candidate[:, :2]
                candidate[:, 0] /= float(W)
                candidate[:, 1] /= float(H)

            bodies = dict(candidate=candidate.tolist(), subset=subset.tolist())
            pose = dict(bodies=bodies, hands=hands, faces=faces)

            canvas = draw_pose(pose, H, W)

        detected_map = HWC3(canvas)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(
            detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

        if return_pil:
            detected_map = Image.fromarray(detected_map)

        return detected_map, candidate, subset


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--chardir',
        type=str,
        default='characters',
        help='Characters root dir.')
    parser.add_argument(
        '--charname', type=str, default='slamdunk', help='Character name.')
    parser.add_argument(
        '--img', type=str, default='resources/sakuragi.png', help='Img name.')
    parser.add_argument(
        '--mask',
        type=str,
        default='resources/sakuragi_mask.png',
        help='Img mask name.')

    args, unknown = parser.parse_known_args()

    return args, unknown


def main():
    args, unknown = parse_args()

    detect_resolution = 512
    control_detector = 'lllyasviel/ControlNet'
    posedet = OpenposeDetectorPoint.from_pretrained(control_detector)

    char_root_dir = args.chardir
    char_name = args.charname
    image_name = args.img
    mask_name = args.mask

    image = Image.open(image_name)
    detected_map, candidate, subset = posedet(image)
    pose_name = 'pose.png'
    detected_map.save(os.path.join(char_root_dir, char_name, pose_name))

    # resize image
    image_np = np.array(image, dtype=np.uint8)
    image_np = HWC3(image_np)
    image_np = resize_image(image_np, detect_resolution)
    image_resized = Image.fromarray(image_np)
    image_resized.save(os.path.join(char_root_dir, char_name, 'texture.png'))

    # resize mask image
    mask = Image.open(mask_name)
    mask_np = np.array(mask, dtype=np.uint8)
    mask_np = HWC3(mask_np)
    mask_np = resize_image(mask_np, detect_resolution)
    image_resized = Image.fromarray(mask_np)
    image_resized.save(os.path.join(char_root_dir, char_name, 'mask.png'))

    W, H = image_resized.size

    config_file = os.path.join(char_root_dir, char_name, 'char_cfg.yaml')
    config_dict = {}
    config_dict['width'] = detected_map.size[0]
    config_dict['height'] = detected_map.size[1]

    # yapf: disable
    kpts = candidate
    kpts[:, 0] = kpts[:, 0] * W
    kpts[:, 1] = kpts[:, 1] * H
    skeleton = []
    skeleton.append({'loc' : [round(x) for x in (kpts[8]+kpts[11])/2], 'name': 'root'          , 'parent': None})       # noqa
    skeleton.append({'loc' : [round(x) for x in (kpts[8]+kpts[11])/2], 'name': 'hip'           , 'parent': 'root'})       # noqa
    skeleton.append({'loc' : [round(x) for x in  kpts[1]            ], 'name': 'torso'         , 'parent': 'hip'})       # noqa
    skeleton.append({'loc' : [round(x) for x in  kpts[0]            ], 'name': 'neck'          , 'parent': 'torso'})       # noqa
    skeleton.append({'loc' : [round(x) for x in  kpts[2]            ], 'name': 'right_shoulder', 'parent': 'torso'})       # noqa
    skeleton.append({'loc' : [round(x) for x in  kpts[3]            ], 'name': 'right_elbow'   , 'parent': 'right_shoulder'})       # noqa
    skeleton.append({'loc' : [round(x) for x in  kpts[4]            ], 'name': 'right_hand'    , 'parent': 'right_elbow'})       # noqa
    skeleton.append({'loc' : [round(x) for x in  kpts[5]            ], 'name': 'left_shoulder' , 'parent': 'torso'})       # noqa
    skeleton.append({'loc' : [round(x) for x in  kpts[6]            ], 'name': 'left_elbow'    , 'parent': 'left_shoulder'})       # noqa
    skeleton.append({'loc' : [round(x) for x in  kpts[7]            ], 'name': 'left_hand'     , 'parent': 'left_elbow'})       # noqa
    skeleton.append({'loc' : [round(x) for x in  kpts[8]            ], 'name': 'right_hip'     , 'parent': 'root'})       # noqa
    skeleton.append({'loc' : [round(x) for x in  kpts[9]            ], 'name': 'right_knee'    , 'parent': 'right_hip'})       # noqa
    skeleton.append({'loc' : [round(x) for x in  kpts[10]           ], 'name': 'right_foot'    , 'parent': 'right_knee'})       # noqa
    skeleton.append({'loc' : [round(x) for x in  kpts[11]           ], 'name': 'left_hip'      , 'parent': 'root'})       # noqa
    skeleton.append({'loc' : [round(x) for x in  kpts[12]           ], 'name': 'left_knee'     , 'parent': 'left_hip'})       # noqa
    skeleton.append({'loc' : [round(x) for x in  kpts[13]           ], 'name': 'left_foot'     , 'parent': 'left_knee'})       # noqa
    # yapf: enable
    config_dict['skeleton'] = skeleton

    with open(config_file, 'w') as file:
        yaml.dump(config_dict, file)


if __name__ == '__main__':
    main()
