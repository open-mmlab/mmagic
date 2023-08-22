# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.parallel import collate, scatter

from mmedit.datasets.pipelines import Compose

try:
    from facexlib.utils.face_restoration_helper import FaceRestoreHelper
    has_facexlib = True
except ImportError:
    has_facexlib = False


def restoration_face_inference(model, img, upscale_factor=1, face_size=1024):
    """Inference image with the model.

    Args:
        model (nn.Module): The loaded model.
        img (str): File path of input image.
        upscale_factor (int, optional): The number of times the input image
            is upsampled. Default: 1.
        face_size (int, optional): The size of the cropped and aligned faces.
            Default: 1024.

    Returns:
        Tensor: The predicted restoration result.
    """
    device = next(model.parameters()).device  # model device

    # build the data pipeline
    if model.cfg.get('demo_pipeline', None):
        test_pipeline = model.cfg.demo_pipeline
    elif model.cfg.get('test_pipeline', None):
        test_pipeline = model.cfg.test_pipeline
    else:
        test_pipeline = model.cfg.val_pipeline

    # remove gt from test_pipeline
    keys_to_remove = ['gt', 'gt_path']
    for key in keys_to_remove:
        for pipeline in list(test_pipeline):
            if 'key' in pipeline and key == pipeline['key']:
                test_pipeline.remove(pipeline)
            if 'keys' in pipeline and key in pipeline['keys']:
                pipeline['keys'].remove(key)
                if len(pipeline['keys']) == 0:
                    test_pipeline.remove(pipeline)
            if 'meta_keys' in pipeline and key in pipeline['meta_keys']:
                pipeline['meta_keys'].remove(key)
    # build the data pipeline
    test_pipeline = Compose(test_pipeline)

    # face helper for detecting and aligning faces
    assert has_facexlib, 'Please install FaceXLib to use the demo.'
    face_helper = FaceRestoreHelper(
        upscale_factor,
        face_size=face_size,
        crop_ratio=(1, 1),
        det_model='retinaface_resnet50',
        template_3points=True,
        save_ext='png',
        device=device)

    face_helper.read_image(img)
    # get face landmarks for each face
    face_helper.get_face_landmarks_5(
        only_center_face=False, eye_dist_threshold=None)
    # align and warp each face
    face_helper.align_warp_face()

    for i, img in enumerate(face_helper.cropped_faces):
        # prepare data
        data = dict(lq=img.astype(np.float32))
        data = test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        if 'cuda' in str(device):
            data = scatter(data, [device])[0]

        with torch.no_grad():
            output = model(test_mode=True, **data)['output']
            output = torch.clamp(output, min=0, max=1)

        output = output.squeeze(0).permute(1, 2, 0)[:, :, [2, 1, 0]]
        output = output.cpu().numpy() * 255  # (0, 255)
        face_helper.add_restored_face(output)

    face_helper.get_inverse_affine(None)
    restored_img = face_helper.paste_faces_to_input_image(upsample_img=None)

    return restored_img
