# This code is referenced from BasicSR with modifications.
# Reference: https://github.com/xinntao/BasicSR/blob/master/basicsr/utils/face_util.py  # noqa
# Original licence: Copyright (c) 2020 xinntao, under the Apache 2.0 license.
import os

import cv2
import dlib
import numpy as np
import torch
from skimage import transform as trans


class FaceHelper:
    """Helper for the face restoration pipeline.

    Args:
        upscale_factor (float, optional): The upsampling factor used to produce
            the output image. Default: 1.
        face_size (int, optional): The size of the cropped face. Default: 1024.
    """

    def __init__(self, upscale_factor=1, face_size=1024):
        self.upscale_factor = upscale_factor
        self.face_size = (face_size, face_size)

        # 5 standard landmarks for FFHQ faces with image size 1024 x 1024
        self.face_template = np.array([[686.77227723, 488.62376238],
                                       [586.77227723, 493.59405941],
                                       [337.91089109, 488.38613861],
                                       [437.95049505, 493.51485149],
                                       [513.58415842, 678.5049505]])
        self.face_template = self.face_template / (1024 // face_size)

        # for estimation the 2D similarity transformation
        self.similarity_trans = trans.SimilarityTransform()

        self.all_landmarks_5 = []
        self.all_landmarks_68 = []
        self.affine_matrices = []
        self.inverse_affine_matrices = []
        self.cropped_faces = []
        self.restored_faces = []

        # self.init_dlib(
        #     'pretrained_models/mmod_human_face_detector-4cb19393.dat',
        #     'pretrained_models/shape_predictor_5_face_landmarks-c4b1e980.dat',
        #     'pretrained_models/shape_predictor_68_face_landmarks-fbdc2cb8.dat')

    def init_dlib(self, detection_path, landmark5_path, landmark68_path):
        """Initialize the dlib detectors and predictors."""

        self.face_detector = dlib.cnn_face_detection_model_v1(detection_path)
        self.shape_predictor_5 = dlib.shape_predictor(landmark5_path)
        self.shape_predictor_68 = dlib.shape_predictor(landmark68_path)

    def read_input_image(self, img_path):
        """Read the input image.

        The loaded image is an numpy array with shape (h, w, c), in the
        order of RGB.
        """
        self.input_img = dlib.load_rgb_image(img_path)

    def detect_faces(self,
                     img_path,
                     upsample_num_times=1,
                     is_keep_only_largest=False):
        """ Detect the faces given an image path.

        Args:
            img_path (str): Image path.
            upsample_num_times (int): The number of 2x upsampling before
                running the face detector. Try improving this number if faces
                cannot be detected. Default: 1.
            is_keep_only_largest (bool): Whether to keep only the largest
                face. Default: False.

        Returns:
            int: Number of detected faces.
        """

        self.read_input_image(img_path)
        det_faces = self.face_detector(self.input_img, upsample_num_times)
        self.det_faces = []
        if len(det_faces) == 0:
            print(f'No face detected for [{img_path}]. Try to increase'
                  'upsample_num_times.')
        else:
            if is_keep_only_largest:
                print('Detect several faces and only keep the largest.')
                face_areas = []
                for i in range(len(det_faces)):
                    face_area = (det_faces[i].rect.right() -
                                 det_faces[i].rect.left()) * (
                                     det_faces[i].rect.bottom() -
                                     det_faces[i].rect.top())
                    face_areas.append(face_area)
                largest_idx = face_areas.index(max(face_areas))
                self.det_faces = [det_faces[largest_idx]]
            else:
                self.det_faces = det_faces

        return len(self.det_faces)

    def get_facial_landmarks_5(self):
        """Get 5 facial landmarks for the cropped faces.

        Returns:
            int: Number of sets of landmarks.
        """

        for face in self.det_faces:
            shape = self.shape_predictor_5(self.input_img, face.rect)
            landmark = np.array([[part.x, part.y] for part in shape.parts()])
            self.all_landmarks_5.append(landmark)

        return len(self.all_landmarks_5)

    def get_facial_landmarks_68(self):
        """Get 68 facial landmarks for cropped faces.

        In this function, there should be at most one face in the
        cropped image.

        Returns:
            int: Number of sets of landmarks.
        """

        num_detected_face = 0
        for idx, face in enumerate(self.cropped_faces):
            # face detection
            det_face = self.face_detector(face, 1)  # TODO: can we remove it?
            if len(det_face) == 0:
                raise AssertionError('Cannot find faces in cropped image '
                                     f'with index {idx}.')
            else:
                if len(det_face) > 1:
                    print('Detect several faces in the cropped face. Use the '
                          'largest one. Note that it will also cause overlap '
                          'during paste_faces_to_input_image.')
                    face_areas = []
                    for i in range(len(det_face)):
                        face_area = (det_face[i].rect.right() -
                                     det_face[i].rect.left()) * (
                                         det_face[i].rect.bottom() -
                                         det_face[i].rect.top())
                        face_areas.append(face_area)
                    largest_idx = face_areas.index(max(face_areas))
                    face_rect = det_face[largest_idx].rect
                else:
                    face_rect = det_face[0].rect
                shape = self.shape_predictor_68(face, face_rect)
                landmark = np.array([[part.x, part.y]
                                     for part in shape.parts()])
                self.all_landmarks_68.append(landmark)
                num_detected_face += 1

        return num_detected_face

    def warp_crop_faces(self,
                        cropped_face_save_path=None,
                        inverse_affine_save_path=None):
        """Get the affine matrices, then warp and cropped the faces.

        Args:
            cropped_face_save_path (str | None): The path to save the cropped
                images. This is useful when you want to pre-process a
                dataset for training. Default: None.
            inverse_affine_save_path (str | None): The path to save the inverse
                affine matrix, which is used to paste the cropped faces back
                to the original image. Default: None.

        Returns:
            list[ndarray]: The list of faces cropped from the original image.

        """
        self.get_facial_landmarks_5()
        for idx, landmark in enumerate(self.all_landmarks_5):
            # use 5 landmarks to get affine matrix
            self.similarity_trans.estimate(landmark, self.face_template)
            affine_matrix = self.similarity_trans.params[0:2, :]
            self.affine_matrices.append(affine_matrix)

            # warp and crop faces
            cropped_face = cv2.warpAffine(self.input_img, affine_matrix,
                                          self.face_size)
            self.cropped_faces.append(cropped_face)

            # save the cropped face
            if cropped_face_save_path is not None:
                path, _ = os.path.splitext(cropped_face_save_path)
                cv2.imwrite(f'{path}_{idx:02d}.png',
                            cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))

            # get inverse affine matrix
            self.similarity_trans.estimate(self.face_template,
                                           landmark * self.upscale_factor)
            inverse_affine = self.similarity_trans.params[0:2, :]
            self.inverse_affine_matrices.append(inverse_affine)

            # save inverse affine matrices
            if inverse_affine_save_path is not None:
                path, _ = os.path.splitext(inverse_affine_save_path)
                torch.save(inverse_affine, f'{path}_{idx:02d}.pth')

        return self.cropped_faces

    def get_cropped_faces(self,
                          img_path,
                          upsample_num_times=1,
                          is_keep_only_largest=False,
                          cropped_face_save_path=None,
                          inverse_affine_save_path=None):

        num_det_faces = self.detect_faces(
            img_path,
            upsample_num_times=upsample_num_times,
            is_keep_only_largest=is_keep_only_largest)

        # warp and crop each face
        if num_det_faces > 0:
            is_faces_detected = True
            cropped_faces = self.warp_crop_faces(cropped_face_save_path,
                                                 inverse_affine_save_path)
        else:
            is_faces_detected = False
            cropped_faces = [self.read_input_image(img_path)]

        return cropped_faces, is_faces_detected

    def add_restored_face(self, face):
        """Add the restored faces to a list, for post-processing.

        Args:
            face (ndarray): The face to be post-process.
        """

        self.restored_faces.append(face)

    def paste_faces_to_input_image(self, save_path=None):
        """Paste the cropped faces back to the image.

        Args:
            save_path (str | None, optional): The image path where the output
                image is saved. If None, the image are not saved.

        Returns:
            ndarray: The final output image.


        """
        # operate in the BGR order
        input_img = cv2.cvtColor(self.input_img, cv2.COLOR_RGB2BGR)

        # determine the output size
        h, w, _ = input_img.shape
        h_up, w_up = h * self.upscale_factor, w * self.upscale_factor

        # simply resize the background
        upsample_img = cv2.resize(input_img, (w_up, h_up))

        # number of inverse affine matrix should equal number of faces
        assert len(self.restored_faces) == len(self.inverse_affine_matrices), (
            'length of restored_faces and affine_matrices are different.')

        for restored_face, inverse_affine in zip(self.restored_faces,
                                                 self.inverse_affine_matrices):
            inv_restored = cv2.warpAffine(restored_face, inverse_affine,
                                          (w_up, h_up))
            mask = np.ones((*self.face_size, 3), dtype=np.float32)
            inv_mask = cv2.warpAffine(mask, inverse_affine, (w_up, h_up))

            # remove the black borders
            inv_mask_erosion = cv2.erode(
                inv_mask,
                np.ones((2 * self.upscale_factor, 2 * self.upscale_factor),
                        np.uint8))
            inv_restored_remove_border = inv_mask_erosion * inv_restored
            total_face_area = np.sum(inv_mask_erosion) // 3

            # compute the fusion edge based on the area of face
            w_edge = int(total_face_area**0.5) // 20
            erosion_radius = w_edge * 2
            inv_mask_center = cv2.erode(
                inv_mask_erosion,
                np.ones((erosion_radius, erosion_radius), np.uint8))
            blur_size = w_edge * 2
            inv_soft_mask = cv2.GaussianBlur(inv_mask_center,
                                             (blur_size + 1, blur_size + 1), 0)
            upsample_img = inv_soft_mask * inv_restored_remove_border + (
                1 - inv_soft_mask) * upsample_img

        if save_path is not None:
            save_path = save_path.replace('.jpg',
                                          '.png').replace('.jpeg', '.png')
            cv2.imwrite(save_path, upsample_img.astype(np.uint8))

        self.clean_all()  # clear the variables

        return upsample_img

    def clean_all(self):
        """Clean all the variables for the next image."""
        self.all_landmarks_5 = []
        self.all_landmarks_68 = []
        self.restored_faces = []
        self.affine_matrices = []
        self.cropped_faces = []
        self.inverse_affine_matrices = []
