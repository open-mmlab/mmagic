# Copyright (c) OpenMMLab. All rights reserved.
import io
import logging
import random

import cv2
import numpy as np

from mmedit.datasets.pipelines import blur_kernels as blur_kernels
from ..registry import PIPELINES

try:
    import av
    has_av = True
except ImportError:
    has_av = False


@PIPELINES.register_module()
class RandomBlur:
    """Apply random blur to the input.

    Modified keys are the attributed specified in "keys".

    Args:
        params (dict): A dictionary specifying the degradation settings.
        keys (list[str]): A list specifying the keys whose values are
            modified.
    """

    def __init__(self, params, keys):
        self.keys = keys
        self.params = params

    def get_kernel(self, num_kernels):
        kernel_type = np.random.choice(
            self.params['kernel_list'], p=self.params['kernel_prob'])
        kernel_size = random.choice(self.params['kernel_size'])

        sigma_x_range = self.params.get('sigma_x', [0, 0])
        sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
        sigma_x_step = self.params.get('sigma_x_step', 0)

        sigma_y_range = self.params.get('sigma_y', [0, 0])
        sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
        sigma_y_step = self.params.get('sigma_y_step', 0)

        rotate_angle_range = self.params.get('rotate_angle', [-np.pi, np.pi])
        rotate_angle = np.random.uniform(rotate_angle_range[0],
                                         rotate_angle_range[1])
        rotate_angle_step = self.params.get('rotate_angle_step', 0)

        beta_gau_range = self.params.get('beta_gaussian', [0.5, 4])
        beta_gau = np.random.uniform(beta_gau_range[0], beta_gau_range[1])
        beta_gau_step = self.params.get('beta_gaussian_step', 0)

        beta_pla_range = self.params.get('beta_plateau', [1, 2])
        beta_pla = np.random.uniform(beta_pla_range[0], beta_pla_range[1])
        beta_pla_step = self.params.get('beta_plateau_step', 0)

        omega_range = self.params.get('omega', None)
        omega_step = self.params.get('omega_step', 0)
        if omega_range is None:  # follow Real-ESRGAN settings if not specified
            if kernel_size < 13:
                omega_range = [np.pi / 3., np.pi]
            else:
                omega_range = [np.pi / 5., np.pi]
        omega = np.random.uniform(omega_range[0], omega_range[1])

        # determine blurring kernel
        kernels = []
        for _ in range(0, num_kernels):
            kernel = blur_kernels.random_mixed_kernels(
                [kernel_type],
                [1],
                kernel_size,
                [sigma_x, sigma_x],
                [sigma_y, sigma_y],
                [rotate_angle, rotate_angle],
                [beta_gau, beta_gau],
                [beta_pla, beta_pla],
                [omega, omega],
                None,
            )
            kernels.append(kernel)

            # update kernel parameters
            sigma_x += np.random.uniform(-sigma_x_step, sigma_x_step)
            sigma_y += np.random.uniform(-sigma_y_step, sigma_y_step)
            rotate_angle += np.random.uniform(-rotate_angle_step,
                                              rotate_angle_step)
            beta_gau += np.random.uniform(-beta_gau_step, beta_gau_step)
            beta_pla += np.random.uniform(-beta_pla_step, beta_pla_step)
            omega += np.random.uniform(-omega_step, omega_step)

            sigma_x = np.clip(sigma_x, sigma_x_range[0], sigma_x_range[1])
            sigma_y = np.clip(sigma_y, sigma_y_range[0], sigma_y_range[1])
            rotate_angle = np.clip(rotate_angle, rotate_angle_range[0],
                                   rotate_angle_range[1])
            beta_gau = np.clip(beta_gau, beta_gau_range[0], beta_gau_range[1])
            beta_pla = np.clip(beta_pla, beta_pla_range[0], beta_pla_range[1])
            omega = np.clip(omega, omega_range[0], omega_range[1])

        return kernels

    def _apply_random_blur(self, imgs):
        is_single_image = False
        if isinstance(imgs, np.ndarray):
            is_single_image = True
            imgs = [imgs]

        # get kernel and blur the input
        kernels = self.get_kernel(num_kernels=len(imgs))
        imgs = [
            cv2.filter2D(img, -1, kernel)
            for img, kernel in zip(imgs, kernels)
        ]

        if is_single_image:
            imgs = imgs[0]

        return imgs

    def __call__(self, results):
        if np.random.uniform() > self.params.get('prob', 1):
            return results

        for key in self.keys:
            results[key] = self._apply_random_blur(results[key])

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(params={self.params}, keys={self.keys})')
        return repr_str


@PIPELINES.register_module()
class RandomResize:
    """Randomly resize the input.

    Modified keys are the attributed specified in "keys".

    Args:
        params (dict): A dictionary specifying the degradation settings.
        keys (list[str]): A list specifying the keys whose values are
            modified.
    """

    def __init__(self, params, keys):
        self.keys = keys
        self.params = params

        self.resize_dict = dict(
            bilinear=cv2.INTER_LINEAR,
            bicubic=cv2.INTER_CUBIC,
            area=cv2.INTER_AREA,
            lanczos=cv2.INTER_LANCZOS4)

    def _random_resize(self, imgs):
        is_single_image = False
        if isinstance(imgs, np.ndarray):
            is_single_image = True
            imgs = [imgs]

        h, w = imgs[0].shape[:2]

        resize_opt = self.params['resize_opt']
        resize_prob = self.params['resize_prob']
        resize_opt = np.random.choice(resize_opt, p=resize_prob).lower()
        if resize_opt not in self.resize_dict:
            raise NotImplementedError(f'resize_opt [{resize_opt}] is not '
                                      'implemented')
        resize_opt = self.resize_dict[resize_opt]

        resize_step = self.params.get('resize_step', 0)

        # determine the target size, if not provided
        target_size = self.params.get('target_size', None)
        if target_size is None:
            resize_mode = np.random.choice(['up', 'down', 'keep'],
                                           p=self.params['resize_mode_prob'])
            resize_scale = self.params['resize_scale']
            if resize_mode == 'up':
                scale_factor = np.random.uniform(1, resize_scale[1])
            elif resize_mode == 'down':
                scale_factor = np.random.uniform(resize_scale[0], 1)
            else:
                scale_factor = 1

            # determine output size
            h_out, w_out = h * scale_factor, w * scale_factor
            if self.params.get('is_size_even', False):
                h_out, w_out = 2 * (h_out // 2), 2 * (w_out // 2)
            target_size = (int(h_out), int(w_out))
        else:
            resize_step = 0

        # resize the input
        if resize_step == 0:  # same target_size for all input images
            outputs = [
                cv2.resize(img, target_size[::-1], interpolation=resize_opt)
                for img in imgs
            ]
        else:  # different target_size for each input image
            outputs = []
            for img in imgs:
                img = cv2.resize(
                    img, target_size[::-1], interpolation=resize_opt)
                outputs.append(img)

                # update scale
                scale_factor += np.random.uniform(-resize_step, resize_step)
                scale_factor = np.clip(scale_factor, resize_scale[0],
                                       resize_scale[1])

                # determine output size
                h_out, w_out = h * scale_factor, w * scale_factor
                if self.params.get('is_size_even', False):
                    h_out, w_out = 2 * (h_out // 2), 2 * (w_out // 2)
                target_size = (int(h_out), int(w_out))

        if is_single_image:
            outputs = outputs[0]

        return outputs

    def __call__(self, results):
        if np.random.uniform() > self.params.get('prob', 1):
            return results

        for key in self.keys:
            results[key] = self._random_resize(results[key])

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(params={self.params}, keys={self.keys})')
        return repr_str


@PIPELINES.register_module()
class RandomNoise:
    """Apply random noise to the input.

    Currently support Gaussian noise and Poisson noise.

    Modified keys are the attributed specified in "keys".

    Args:
        params (dict): A dictionary specifying the degradation settings.
        keys (list[str]): A list specifying the keys whose values are
            modified.
    """

    def __init__(self, params, keys):
        self.keys = keys
        self.params = params

    def _apply_gaussian_noise(self, imgs):
        sigma_range = self.params['gaussian_sigma']
        sigma = np.random.uniform(sigma_range[0], sigma_range[1]) / 255.

        sigma_step = self.params.get('gaussian_sigma_step', 0)

        gray_noise_prob = self.params['gaussian_gray_noise_prob']
        is_gray_noise = np.random.uniform() < gray_noise_prob

        outputs = []
        for img in imgs:
            noise = np.float32(np.random.randn(*(img.shape))) * sigma
            if is_gray_noise:
                noise = noise[:, :, :1]
            outputs.append(img + noise)

            # update noise level
            sigma += np.random.uniform(-sigma_step, sigma_step) / 255.
            sigma = np.clip(sigma, sigma_range[0] / 255.,
                            sigma_range[1] / 255.)

        return outputs

    def _apply_poisson_noise(self, imgs):
        scale_range = self.params['poisson_scale']
        scale = np.random.uniform(scale_range[0], scale_range[1])

        scale_step = self.params.get('poisson_scale_step', 0)

        gray_noise_prob = self.params['poisson_gray_noise_prob']
        is_gray_noise = np.random.uniform() < gray_noise_prob

        outputs = []
        for img in imgs:
            noise = img.copy()
            if is_gray_noise:
                noise = cv2.cvtColor(noise[..., [2, 1, 0]], cv2.COLOR_BGR2GRAY)
                noise = noise[..., np.newaxis]
            noise = np.clip((noise * 255.0).round(), 0, 255) / 255.
            unique_val = 2**np.ceil(np.log2(len(np.unique(noise))))
            noise = np.random.poisson(noise * unique_val) / unique_val - noise

            outputs.append(img + noise * scale)

            # update noise level
            scale += np.random.uniform(-scale_step, scale_step)
            scale = np.clip(scale, scale_range[0], scale_range[1])

        return outputs

    def _apply_random_noise(self, imgs):
        noise_type = np.random.choice(
            self.params['noise_type'], p=self.params['noise_prob'])

        is_single_image = False
        if isinstance(imgs, np.ndarray):
            is_single_image = True
            imgs = [imgs]

        if noise_type.lower() == 'gaussian':
            imgs = self._apply_gaussian_noise(imgs)
        elif noise_type.lower() == 'poisson':
            imgs = self._apply_poisson_noise(imgs)
        else:
            raise NotImplementedError(f'"noise_type" [{noise_type}] is '
                                      'not implemented.')

        if is_single_image:
            imgs = imgs[0]

        return imgs

    def __call__(self, results):
        if np.random.uniform() > self.params.get('prob', 1):
            return results

        for key in self.keys:
            results[key] = self._apply_random_noise(results[key])

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(params={self.params}, keys={self.keys})')
        return repr_str


@PIPELINES.register_module()
class RandomJPEGCompression:
    """Apply random JPEG compression to the input.

    Modified keys are the attributed specified in "keys".

    Args:
        params (dict): A dictionary specifying the degradation settings.
        keys (list[str]): A list specifying the keys whose values are
            modified.
    """

    def __init__(self, params, keys):
        self.keys = keys
        self.params = params

    def _apply_random_compression(self, imgs):
        is_single_image = False
        if isinstance(imgs, np.ndarray):
            is_single_image = True
            imgs = [imgs]

        # determine initial compression level and the step size
        quality = self.params['quality']
        quality_step = self.params.get('quality_step', 0)
        jpeg_param = round(np.random.uniform(quality[0], quality[1]))

        # apply jpeg compression
        outputs = []
        for img in imgs:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_param]
            _, img_encoded = cv2.imencode('.jpg', img * 255., encode_param)
            outputs.append(np.float32(cv2.imdecode(img_encoded, 1)) / 255.)

            # update compression level
            jpeg_param += np.random.uniform(-quality_step, quality_step)
            jpeg_param = round(np.clip(jpeg_param, quality[0], quality[1]))

        if is_single_image:
            outputs = outputs[0]

        return outputs

    def __call__(self, results):
        if np.random.uniform() > self.params.get('prob', 1):
            return results

        for key in self.keys:
            results[key] = self._apply_random_compression(results[key])

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(params={self.params}, keys={self.keys})')
        return repr_str


@PIPELINES.register_module()
class RandomVideoCompression:
    """Apply random video compression to the input.

    Modified keys are the attributed specified in "keys".

    Args:
        params (dict): A dictionary specifying the degradation settings.
        keys (list[str]): A list specifying the keys whose values are
            modified.
    """

    def __init__(self, params, keys):
        assert has_av, 'Please install av to use video compression.'

        self.keys = keys
        self.params = params
        logging.getLogger('libav').setLevel(50)

    def _apply_random_compression(self, imgs):
        codec = random.choices(self.params['codec'],
                               self.params['codec_prob'])[0]
        bitrate = self.params['bitrate']
        bitrate = np.random.randint(bitrate[0], bitrate[1] + 1)

        buf = io.BytesIO()
        with av.open(buf, 'w', 'mp4') as container:
            stream = container.add_stream(codec, rate=1)
            stream.height = imgs[0].shape[0]
            stream.width = imgs[0].shape[1]
            stream.pix_fmt = 'yuv420p'
            stream.bit_rate = bitrate

            for img in imgs:
                img = (255 * img).astype(np.uint8)
                frame = av.VideoFrame.from_ndarray(img, format='rgb24')
                frame.pict_type = 'NONE'
                for packet in stream.encode(frame):
                    container.mux(packet)

            # Flush stream
            for packet in stream.encode():
                container.mux(packet)

        outputs = []
        with av.open(buf, 'r', 'mp4') as container:
            if container.streams.video:
                for frame in container.decode(**{'video': 0}):
                    outputs.append(
                        frame.to_rgb().to_ndarray().astype(np.float32) / 255.)

        return outputs

    def __call__(self, results):
        if np.random.uniform() > self.params.get('prob', 1):
            return results

        for key in self.keys:
            results[key] = self._apply_random_compression(results[key])

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(params={self.params}, keys={self.keys})')
        return repr_str


allowed_degradations = {
    'RandomBlur': RandomBlur,
    'RandomResize': RandomResize,
    'RandomNoise': RandomNoise,
    'RandomJPEGCompression': RandomJPEGCompression,
    'RandomVideoCompression': RandomVideoCompression,
}


@PIPELINES.register_module()
class DegradationsWithShuffle:
    """Apply random degradations to input, with degradations being shuffled.

    Degradation groups are supported. The order of degradations within the same
    group is preserved. For example, if we have degradations = [a, b, [c, d]]
    and shuffle_idx = None, then the possible orders are

    ::

        [a, b, [c, d]]
        [a, [c, d], b]
        [b, a, [c, d]]
        [b, [c, d], a]
        [[c, d], a, b]
        [[c, d], b, a]

    Modified keys are the attributed specified in "keys".

    Args:
        degradations (list[dict]): The list of degradations.
        keys (list[str]): A list specifying the keys whose values are
            modified.
        shuffle_idx (list | None, optional): The degradations corresponding to
            these indices are shuffled. If None, all degradations are shuffled.
    """

    def __init__(self, degradations, keys, shuffle_idx=None):

        self.keys = keys

        self.degradations = self._build_degradations(degradations)

        if shuffle_idx is None:
            self.shuffle_idx = list(range(0, len(degradations)))
        else:
            self.shuffle_idx = shuffle_idx

    def _build_degradations(self, degradations):
        for i, degradation in enumerate(degradations):
            if isinstance(degradation, (list, tuple)):
                degradations[i] = self._build_degradations(degradation)
            else:
                degradation_ = allowed_degradations[degradation['type']]
                degradations[i] = degradation_(degradation['params'],
                                               self.keys)

        return degradations

    def __call__(self, results):
        # shuffle degradations
        if len(self.shuffle_idx) > 0:
            shuffle_list = [self.degradations[i] for i in self.shuffle_idx]
            np.random.shuffle(shuffle_list)
            for i, idx in enumerate(self.shuffle_idx):
                self.degradations[idx] = shuffle_list[i]

        # apply degradations to input
        for degradation in self.degradations:
            if isinstance(degradation, (tuple, list)):
                for subdegrdation in degradation:
                    results = subdegrdation(results)
            else:
                results = degradation(results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(degradations={self.degradations}, '
                     f'keys={self.keys}, '
                     f'shuffle_idx={self.shuffle_idx})')
        return repr_str
