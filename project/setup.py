from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    ext_modules=[
        CUDAExtension(
            name="ops.my_gelu_op",
            sources=[
                "ops/binding.cpp",
                "ops/gelu_and_mul.cu",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
