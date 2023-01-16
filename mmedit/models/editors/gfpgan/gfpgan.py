# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import torch
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from torchvision.transforms.functional import normalize

from mmedit.registry import MODELS
from ...base_models import BaseGAN
from .gfpgan_utils import img2tensor, tensor2img


@MODELS.register_module('GFPGAN')
class GFPGAN(BaseGAN):

    def __init__(self,
                 bg_upsampler=None,
                 face_restore_cfg=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.bg_upsampler = bg_upsampler

        # initialize face helper
        if face_restore_cfg is not None:
            self.upscale = face_restore_cfg['upscale']
            self.face_helper = FaceRestoreHelper(
                self.upscale,
                face_size=512,
                crop_ratio=(1, 1),
                det_model='retinaface_resnet50',
                save_ext='png',
                use_parse=True,
                device=face_restore_cfg['device'],
                model_rootpath='gfpgan/weights')
        else:
            self.face_helper = None

    @torch.no_grad()
    def enhance(self,
                img,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
                weight=0.5):
        self.face_helper.clean_all()

        if has_aligned:  # the inputs are already aligned
            img = cv2.resize(img, (512, 512))
            self.face_helper.cropped_faces = [img]
        else:
            self.face_helper.read_image(img)
            # get face landmarks for each face
            self.face_helper.get_face_landmarks_5(
                only_center_face=only_center_face, eye_dist_threshold=5)

            self.face_helper.align_warp_face()

        # face restoration
        for cropped_face in self.face_helper.cropped_faces:
            # prepare data
            cropped_face_t = img2tensor(
                cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(
                cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

            output = self.generator(
                cropped_face_t, return_rgb=False, weight=weight)[0]
            restored_face = tensor2img(
                output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))

            restored_face = restored_face.astype('uint8')
            self.face_helper.add_restored_face(restored_face)

        if not has_aligned and paste_back:
            # upsample the background
            if self.bg_upsampler is not None:
                # Now only support RealESRGAN for upsampling background
                bg_img = self.bg_upsampler.enhance(
                    img, outscale=self.upscale)[0]
            else:
                bg_img = None

            self.face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            restored_img = self.face_helper.paste_faces_to_input_image(
                upsample_img=bg_img)
            return self.face_helper.cropped_faces, \
                self.face_helper.restored_faces, restored_img
        else:
            return self.face_helper.cropped_faces, \
                self.face_helper.restored_faces, None
