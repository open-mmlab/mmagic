import cv2
import numpy as np

from ..registry import PIPELINES


@PIPELINES.register_module
class MergeFgAndBg(object):
    """Composite foreground image and background image with alpha.

    Required keys are "alpha", "fg" and "bg", added key is "merged".
    """

    def __call__(self, results):
        alpha = results['alpha'][..., None].astype(np.float32) / 255.
        fg = results['fg']
        bg = results['bg']
        merged = fg * alpha + (1. - alpha) * bg
        results['merged'] = merged
        return results


@PIPELINES.register_module
class GenerateTrimap(object):
    """Using random erode/dilate to generate trimap from alpha matte.

    Required key is "alpha", added key is "trimap".

    Args:
        kernel_size (int | tuple[int]): the range of random kernel_size of
            erode/dilate; int indicates a fixed kernel_size.
        iterations (int | tuple[int]): the range of random iterations of
            erode/dilate; int indicates a fixed iterations.
        symmetric (bool): wether use the same kernel_size and iterations for
            both erode and dilate.
    """

    def __init__(self, kernel_size, iterations=1, symmetric=False):
        if isinstance(kernel_size, int):
            min_kernel, max_kernel = kernel_size, kernel_size + 1
        else:
            min_kernel, max_kernel = kernel_size

        if isinstance(iterations, int):
            self.min_iteration, self.max_iteration = iterations, iterations + 1
        else:
            self.min_iteration, self.max_iteration = iterations

        self.kernels = [
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            for size in range(min_kernel, max_kernel)
        ]
        self.symmetric = symmetric

    def __call__(self, results):
        alpha = results['alpha']

        kernel_num = len(self.kernels)
        erode_ksize_idx = np.random.randint(kernel_num)
        erode_iter = np.random.randint(self.min_iteration, self.max_iteration)
        if self.symmetric:
            dilate_ksize_idx = erode_ksize_idx
            dilate_iter = erode_iter
        else:
            dilate_ksize_idx = np.random.randint(kernel_num)
            dilate_iter = np.random.randint(self.min_iteration,
                                            self.max_iteration)

        eroded = cv2.erode(
            alpha, self.kernels[erode_ksize_idx], iterations=erode_iter)
        dilated = cv2.dilate(
            alpha, self.kernels[dilate_ksize_idx], iterations=dilate_iter)

        trimap = np.zeros_like(alpha)
        trimap.fill(128)
        trimap[eroded >= 255] = 255
        trimap[dilated <= 0] = 0
        results['trimap'] = trimap.astype(np.float32)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f'(kernels={self.kernels}, min_iteration={self.min_iteration}, '
            f'max_iteration={self.max_iteration}, symmetric={self.symmetric})')
        return repr_str
