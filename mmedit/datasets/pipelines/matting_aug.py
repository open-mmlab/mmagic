import os.path as osp
import random

import cv2
import mmcv
import numpy as np

from ..registry import PIPELINES
from .utils import adjust_gamma, random_choose_unknown


def add_gaussian_noise(img, mu, sigma):
    img = img.astype(np.float32)
    gauss_noise = np.random.normal(mu, sigma, img.shape)
    noisy_img = img + gauss_noise
    noisy_img = np.clip(noisy_img, 0, 255)
    return noisy_img


@PIPELINES.register_module()
class MergeFgAndBg(object):
    """Composite foreground image and background image with alpha.

    Required keys are "alpha", "fg" and "bg", added key is "merged".
    """

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        alpha = results['alpha'][..., None].astype(np.float32) / 255.
        fg = results['fg']
        bg = results['bg']
        merged = fg * alpha + (1. - alpha) * bg
        results['merged'] = merged
        return results


@PIPELINES.register_module()
class GenerateTrimap(object):
    """Using random erode/dilate to generate trimap from alpha matte.

    Required key is "alpha", added key is "trimap".

    Args:
        kernel_size (int | tuple[int]): The range of random kernel_size of
            erode/dilate; int indicates a fixed kernel_size. If `random` is set
            to False and kernel_size is a tuple of length 2, then it will be
            interpreted as (erode kernel_size, dilate kernel_size). It should
            be noted that the kernel of the erosion and dilation has the same
            height and width.
        iterations (int | tuple[int], optional): The range of random iterations
            of erode/dilate; int indicates a fixed iterations. If `random` is
            set to False and iterations is a tuple of length 2, then it will be
            interpreted as (erode iterations, dilate iterations). Default to 1.
        random (bool, optional): Whether use random kernel_size and iterations
            when generating trimap. See `kernel_size` and `iterations` for more
            information.
    """

    def __init__(self, kernel_size, iterations=1, random=True):
        if isinstance(kernel_size, int):
            kernel_size = kernel_size, kernel_size + 1
        elif not mmcv.is_tuple_of(kernel_size, int) or len(kernel_size) != 2:
            raise ValueError('kernel_size must be an int or a tuple of 2 int, '
                             f'but got {kernel_size}')

        if isinstance(iterations, int):
            iterations = iterations, iterations + 1
        elif not mmcv.is_tuple_of(iterations, int) or len(iterations) != 2:
            raise ValueError('iterations must be an int or a tuple of 2 int, '
                             f'but got {iterations}')

        self.random = random
        if self.random:
            min_kernel, max_kernel = kernel_size
            self.iterations = iterations
            self.kernels = [
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
                for size in range(min_kernel, max_kernel)
            ]
        else:
            erode_ksize, dilate_ksize = kernel_size
            self.iterations = iterations
            self.kernels = [
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (erode_ksize, erode_ksize)),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (dilate_ksize, dilate_ksize))
            ]

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        alpha = results['alpha']

        if self.random:
            kernel_num = len(self.kernels)
            erode_kernel_idx = np.random.randint(kernel_num)
            dilate_kernel_idx = np.random.randint(kernel_num)
            min_iter, max_iter = self.iterations
            erode_iter = np.random.randint(min_iter, max_iter)
            dilate_iter = np.random.randint(min_iter, max_iter)
        else:
            erode_kernel_idx, dilate_kernel_idx = 0, 1
            erode_iter, dilate_iter = self.iterations

        eroded = cv2.erode(
            alpha, self.kernels[erode_kernel_idx], iterations=erode_iter)
        dilated = cv2.dilate(
            alpha, self.kernels[dilate_kernel_idx], iterations=dilate_iter)

        trimap = np.zeros_like(alpha)
        trimap.fill(128)
        trimap[eroded >= 255] = 255
        trimap[dilated <= 0] = 0
        results['trimap'] = trimap.astype(np.float32)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(kernels={self.kernels}, iterations={self.iterations}, '
                     f'random={self.random})')
        return repr_str


@PIPELINES.register_module()
class GenerateTrimapWithDistTransform(object):
    """Generate trimap with distance transform function.

    Args:
        dist_thr (int, optional): Distance threshold. Area with alpha value
            between (0, 255) will be considered as initial unknown area. Then
            area with distance to unknown area smaller than the distance
            threshold will also be consider as unknown area. Defaults to 20.
        random (bool, optional): If True, use random distance threshold from
            [1, dist_thr). If False, use `dist_thr` as the distance threshold
            directly. Defaults to True.
    """

    def __init__(self, dist_thr=20, random=True):
        if not (isinstance(dist_thr, int) and dist_thr >= 1):
            raise ValueError('dist_thr must be an int that is greater than 1, '
                             f'but got {dist_thr}')
        self.dist_thr = dist_thr
        self.random = random

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        alpha = results['alpha']

        # image dilation implemented by Euclidean distance transform
        known = (alpha == 0) | (alpha == 255)
        dist_to_unknown = cv2.distanceTransform(
            known.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        dist_thr = np.random.randint(
            1, self.dist_thr) if self.random else self.dist_thr
        unknown = dist_to_unknown <= dist_thr

        trimap = (alpha == 255) * 255
        trimap[unknown] = 128
        results['trimap'] = trimap.astype(np.uint8)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(dist_thr={self.dist_thr}, random={self.random})'
        return repr_str


@PIPELINES.register_module()
class CompositeFg(object):
    """Composite foreground with a random foreground.

    This class composites the current training sample with additional data
    randomly (could be from the same dataset). With probability 0.5, the sample
    will be composited with a random sample from the specified directory.
    The composition is performed as:

    .. math::
        fg_{new} = \\alpha_1 * fg_1 + (1 - \\alpha_1) * fg_2

        \\alpha_{new} = 1 - (1 - \\alpha_1) * (1 - \\alpha_2)

    where :math:`(fg_1, \\alpha_1)` is from the current sample and
    :math:`(fg_2, \\alpha_2)` is the randomly loaded sample. With the above
    composition, :math:`\\alpha_{new}` is still in `[0, 1]`.

    Required keys are "alpha" and "fg". Modified keys are "alpha" and "fg".

    Args:
        fg_dirs (str | list[str]): Path of directories to load foreground
            images from.
        alpha_dirs (str | list[str]): Path of directories to load alpha mattes
            from.
        interpolation (str): Interpolation method of `mmcv.imresize` to resize
            the randomly loaded images.
    """

    def __init__(self, fg_dirs, alpha_dirs, interpolation='nearest'):
        self.fg_dirs = fg_dirs if isinstance(fg_dirs, list) else [fg_dirs]
        self.alpha_dirs = alpha_dirs if isinstance(alpha_dirs,
                                                   list) else [alpha_dirs]
        self.interpolation = interpolation

        self.fg_list, self.alpha_list = self._get_file_list(
            self.fg_dirs, self.alpha_dirs)

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        fg = results['fg']
        alpha = results['alpha'].astype(np.float32) / 255.
        h, w = results['fg'].shape[:2]

        # randomly select fg
        if np.random.rand() < 0.5:
            idx = np.random.randint(len(self.fg_list))
            fg2 = mmcv.imread(self.fg_list[idx])
            alpha2 = mmcv.imread(self.alpha_list[idx], 'grayscale')
            alpha2 = alpha2.astype(np.float32) / 255.

            fg2 = mmcv.imresize(fg2, (w, h), interpolation=self.interpolation)
            alpha2 = mmcv.imresize(
                alpha2, (w, h), interpolation=self.interpolation)

            # the overlap of two 50% transparency will be 75%
            alpha_tmp = 1 - (1 - alpha) * (1 - alpha2)
            # if the result alpha is all-one, then we avoid composition
            if np.any(alpha_tmp < 1):
                # composite fg with fg2
                fg = fg.astype(np.float32) * alpha[..., None] \
                     + fg2.astype(np.float32) * (1 - alpha[..., None])
                alpha = alpha_tmp
                fg.astype(np.uint8)

        results['fg'] = fg
        results['alpha'] = (alpha * 255).astype(np.uint8)
        return results

    @staticmethod
    def _get_file_list(fg_dirs, alpha_dirs):
        all_fg_list = list()
        all_alpha_list = list()
        for fg_dir, alpha_dir in zip(fg_dirs, alpha_dirs):
            fg_list = sorted(mmcv.scandir(fg_dir))
            alpha_list = sorted(mmcv.scandir(alpha_dir))
            # we assume the file names for fg and alpha are the same
            assert len(fg_list) == len(alpha_list), (
                f'{fg_dir} and {alpha_dir} should have the same number of '
                f'images ({len(fg_list)} differs from ({len(alpha_list)})')
            fg_list = [osp.join(fg_dir, fg) for fg in fg_list]
            alpha_list = [osp.join(alpha_dir, alpha) for alpha in alpha_list]

            all_fg_list.extend(fg_list)
            all_alpha_list.extend(alpha_list)
        return all_fg_list, all_alpha_list

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(fg_dirs={self.fg_dirs}, alpha_dirs={self.alpha_dirs}, '
                     f"interpolation='{self.interpolation}')")
        return repr_str


@PIPELINES.register_module()
class GenerateSeg(object):
    """Generate segmentation mask from alpha matte.

    Args:
        kernel_size (int, optional): Kernel size for both erosion and
            dilation. The kernel will have the same height and width.
            Defaults to 5.
        erode_iter_range (tuple, optional): Iteration of erosion.
            Defaults to (10, 20).
        dilate_iter_range (tuple, optional): Iteration of dilation.
            Defaults to (15, 30).
        num_holes_range (tuple, optional): Range of number of holes to
            randomly select from. Defaults to (0, 3).
        hole_sizes (list, optional): List of (h, w) to be selected as the
            size of the rectangle hole.
            Defaults to [(15, 15), (25, 25), (35, 35), (45, 45)].
        blur_ksizes (list, optional): List of (h, w) to be selected as the
            kernel_size of the gaussian blur.
            Defaults to [(21, 21), (31, 31), (41, 41)].
    """

    def __init__(self,
                 kernel_size=5,
                 erode_iter_range=(10, 20),
                 dilate_iter_range=(15, 30),
                 num_holes_range=(0, 3),
                 hole_sizes=[(15, 15), (25, 25), (35, 35), (45, 45)],
                 blur_ksizes=[(21, 21), (31, 31), (41, 41)]):
        self.kernel_size = kernel_size
        self.erode_iter_range = erode_iter_range
        self.dilate_iter_range = dilate_iter_range
        self.num_holes_range = num_holes_range
        self.hole_sizes = hole_sizes
        self.blur_ksizes = blur_ksizes

    @staticmethod
    def _crop_hole(img, start_point, hole_size):
        """Create a all-zero rectangle hole in the image.

        Args:
            img (np.ndarray): Source image.
            start_point (tuple[int]): The top-left point of the rectangle.
            hole_size (tuple[int]): The height and width of the rectangle hole.

        Return:
            np.ndarray: The cropped image.
        """
        top, left = start_point
        bottom = top + hole_size[0]
        right = left + hole_size[1]
        height, weight = img.shape[:2]
        if top < 0 or bottom > height or left < 0 or right > weight:
            raise ValueError(f'crop area {(left, top, right, bottom)} exceeds '
                             f'image size {(height, weight)}')
        img[top:bottom, left:right] = 0
        return img

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        alpha = results['alpha']
        trimap = results['trimap']

        # generete segmentation mask
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size))
        seg = (alpha > 0.5).astype(np.float32)
        seg = cv2.erode(
            seg, kernel, iterations=np.random.randint(*self.erode_iter_range))
        seg = cv2.dilate(
            seg, kernel, iterations=np.random.randint(*self.dilate_iter_range))

        # generate some holes in segmentation mask
        num_holes = np.random.randint(*self.num_holes_range)
        for i in range(num_holes):
            hole_size = random.choice(self.hole_sizes)
            unknown = trimap == 128
            start_point = random_choose_unknown(unknown, hole_size)
            seg = self._crop_hole(seg, start_point, hole_size)
            trimap = self._crop_hole(trimap, start_point, hole_size)

        # perform gaussian blur to segmentation mask
        seg = cv2.GaussianBlur(seg, random.choice(self.blur_ksizes), 0)

        results['seg'] = seg.astype(np.uint8)
        results['num_holes'] = num_holes
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f'(kernel_size={self.kernel_size}, '
            f'erode_iter_range={self.erode_iter_range}, '
            f'dilate_iter_range={self.dilate_iter_range}, '
            f'num_holes_range={self.num_holes_range}, '
            f'hole_sizes={self.hole_sizes}, blur_ksizes={self.blur_ksizes}')
        return repr_str


@PIPELINES.register_module()
class PerturbBg(object):
    """Randomly add gaussian noise or gamma change to background image.

    Required key is "bg", added key is "noisy_bg".

    Args:
        gamma_ratio (float, optional): The probability to use gamma correction
            instead of gaussian noise. Defaults to 0.6.
    """

    def __init__(self, gamma_ratio=0.6):
        if gamma_ratio < 0 or gamma_ratio > 1:
            raise ValueError('gamma_ratio must be a float between [0, 1], '
                             f'but got {gamma_ratio}')
        self.gamma_ratio = gamma_ratio

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        if np.random.rand() >= self.gamma_ratio:
            # generate gaussian noise with random guassian N([-7, 7), [2, 6))
            mu = np.random.randint(-7, 7)
            sigma = np.random.randint(2, 6)
            results['noisy_bg'] = add_gaussian_noise(results['bg'], mu, sigma)
        else:
            # adjust gamma in a range of N(1, 0.12)
            gamma = np.random.normal(1, 0.12)
            results['noisy_bg'] = adjust_gamma(results['bg'], gamma)
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(gamma_ratio={self.gamma_ratio})'


@PIPELINES.register_module()
class GenerateSoftSeg(object):
    """Generate soft segmentation mask from input segmentation mask.

    Required key is "seg", added key is "soft_seg".

    Args:
        fg_thr (float, optional): Threhold of the foreground in the normalized
            input segmentation mask. Defaults to 0.2.
        border_width (int, optional): Width of border to be padded to the
            bottom of the mask. Defaults to 25.
        erode_ksize (int, optional): Fixed kernel size of the erosion.
            Defaults to 5.
        dilate_ksize (int, optional): Fixed kernel size of the dilation.
            Defaults to 5.
        erode_iter_range (tuple, optional): Iteration of erosion.
            Defaults to (10, 20).
        dilate_iter_range (tuple, optional): Iteration of dilation.
            Defaults to (3, 7).
        blur_ksizes (list, optional): List of (h, w) to be selected as the
            kernel_size of the gaussian blur.
            Defaults to [(21, 21), (31, 31), (41, 41)].
    """

    def __init__(self,
                 fg_thr=0.2,
                 border_width=25,
                 erode_ksize=3,
                 dilate_ksize=5,
                 erode_iter_range=(10, 20),
                 dilate_iter_range=(3, 7),
                 blur_ksizes=[(21, 21), (31, 31), (41, 41)]):
        if not isinstance(fg_thr, float):
            raise TypeError(f'fg_thr must be a float, but got {type(fg_thr)}')
        if not isinstance(border_width, int):
            raise TypeError(
                f'border_width must be an int, but got {type(border_width)}')
        if not isinstance(erode_ksize, int):
            raise TypeError(
                f'erode_ksize must be an int, but got {type(erode_ksize)}')
        if not isinstance(dilate_ksize, int):
            raise TypeError(
                f'dilate_ksize must be an int, but got {type(dilate_ksize)}')
        if (not mmcv.is_tuple_of(erode_iter_range, int)
                or len(erode_iter_range) != 2):
            raise TypeError('erode_iter_range must be a tuple of 2 int, '
                            f'but got {erode_iter_range}')
        if (not mmcv.is_tuple_of(dilate_iter_range, int)
                or len(dilate_iter_range) != 2):
            raise TypeError('dilate_iter_range must be a tuple of 2 int, '
                            f'but got {dilate_iter_range}')
        if not mmcv.is_list_of(blur_ksizes, tuple):
            raise TypeError(
                f'blur_ksizes must be a list of tuple, but got {blur_ksizes}')

        self.fg_thr = fg_thr
        self.border_width = border_width
        self.erode_ksize = erode_ksize
        self.dilate_ksize = dilate_ksize
        self.erode_iter_range = erode_iter_range
        self.dilate_iter_range = dilate_iter_range
        self.blur_ksizes = blur_ksizes

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        seg = results['seg'].astype(np.float32) / 255
        height, width = seg.shape[:2]
        seg[seg > self.fg_thr] = 1

        # to align with the original repo, pad the bottom of the mask
        seg = cv2.copyMakeBorder(seg, 0, self.border_width, 0, 0,
                                 cv2.BORDER_REPLICATE)

        # erode/dilate segmentation mask
        erode_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.erode_ksize, self.erode_ksize))
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.dilate_ksize, self.dilate_ksize))
        seg = cv2.erode(
            seg,
            erode_kernel,
            iterations=np.random.randint(*self.erode_iter_range))
        seg = cv2.dilate(
            seg,
            dilate_kernel,
            iterations=np.random.randint(*self.dilate_iter_range))

        # perform gaussian blur to segmentation mask
        seg = cv2.GaussianBlur(seg, random.choice(self.blur_ksizes), 0)

        # remove the padded rows
        seg = (seg * 255).astype(np.uint8)
        seg = np.delete(seg, range(height, height + self.border_width), 0)

        results['soft_seg'] = seg
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(fg_thr={self.fg_thr}, '
                     f'border_width={self.border_width}, '
                     f'erode_ksize={self.erode_ksize}, '
                     f'dilate_ksize={self.dilate_ksize}, '
                     f'erode_iter_range={self.erode_iter_range}, '
                     f'dilate_iter_range={self.dilate_iter_range}, '
                     f'blur_ksizes={self.blur_ksizes})')
        return repr_str
