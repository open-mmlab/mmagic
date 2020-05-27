from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

NVCC_ARGS = [
    '-D__CUDA_NO_HALF_OPERATORS__',
    '-D__CUDA_NO_HALF_CONVERSIONS__',
    '-D__CUDA_NO_HALF2_OPERATORS__',
]

setup(
    name='carafe',
    ext_modules=[
        CUDAExtension(
            'carafe_ext', [
                'src/cuda/carafe_cuda.cpp', 'src/cuda/carafe_cuda_kernel.cu',
                'src/carafe_ext.cpp'
            ],
            extra_compile_args={
                'cxx': [],
                'nvcc': NVCC_ARGS
            })
    ],
    cmdclass={'build_ext': BuildExtension})
