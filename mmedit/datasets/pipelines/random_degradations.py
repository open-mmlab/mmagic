import random

import cv2
import numpy as np

from mmedit.datasets.pipelines import blur_kernels as blur_kernels
from ..registry import PIPELINES


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

    def get_kernel(self):
        kernel = np.random.choice(
            self.params['kernel_list'], p=self.params['kernel_prob'])
        kernel_size = random.choice(self.params['kernel_size'])

        sigma_x = self.params.get('sigma_x', [0, 0])
        sigma_x = np.random.uniform(sigma_x[0], sigma_x[1])

        sigma_y = self.params.get('sigma_y', [0, 0])
        sigma_y = np.random.uniform(sigma_y[0], sigma_y[1])

        rotate_angle = self.params.get('rotate_angle', [-np.pi, np.pi])
        rotate_angle = np.random.uniform(rotate_angle[0], rotate_angle[1])

        beta_gau = self.params.get('beta_gaussian', [0.5, 4])
        beta_gau = np.random.uniform(beta_gau[0], beta_gau[1])

        beta_pla = self.params.get('beta_plateau', [1, 2])
        beta_pla = np.random.uniform(beta_pla[0], beta_pla[1])

        omega = self.params.get('omega', None)
        if omega is not None:
            omega = np.random.uniform(omega[0], omega[1])
        else:  # follow Real-ESRGAN
            if kernel_size < 13:
                omega = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega = np.random.uniform(np.pi / 5, np.pi)

        # determine blurring kernel
        kernel = blur_kernels.random_mixed_kernels(
            [kernel],
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

        return kernel

    def _apply_random_blur(self, imgs):
        is_single_image = False
        if isinstance(imgs, np.ndarray):
            is_single_image = True
            imgs = [imgs]

        # get kernel and blur the input
        kernel = self.get_kernel()
        imgs = [cv2.filter2D(im, -1, kernel) for im in imgs]

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
            target_size = (int(h * scale_factor), int(w * scale_factor))
        # resize the input
        imgs = [
            cv2.resize(img, target_size[::-1], interpolation=resize_opt)
            for img in imgs
        ]

        if is_single_image:
            imgs = imgs[0]

        return imgs

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

        gray_noise_prob = self.params['gaussian_gray_noise_prob']
        is_gray_noise = np.random.uniform() < gray_noise_prob

        outputs = []
        for img in imgs:
            noise = np.float32(np.random.randn(*(img.shape))) * sigma
            if is_gray_noise:
                noise = noise[:, :, :1]
            outputs.append(img + noise)

        return outputs

    def _apply_poisson_noise(self, imgs):
        scale_range = self.params['poisson_scale']
        scale = np.random.uniform(scale_range[0], scale_range[1])

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

        # determine compression level
        quality = self.params['quality']
        jpeg_param = round(np.random.uniform(quality[0], quality[1]))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_param]

        # apply jpeg compression
        outputs = []
        for img in imgs:
            _, img_encoded = cv2.imencode('.jpg', img * 255., encode_param)
            outputs.append(np.float32(cv2.imdecode(img_encoded, 1)) / 255.)

        imgs = outputs

        if is_single_image:
            imgs = imgs[0]

        return imgs

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
