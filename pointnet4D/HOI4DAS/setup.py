"""
Build script for the pn4d package.

This file only configures the C++/CUDA extension. All other metadata
(name, version, dependencies, …) is declared in pyproject.toml.

The extension is registered as ``pn4d._C`` so that it ships inside the
package and avoids any clash with the legacy ``pointnet2._ext`` build.
"""
from __future__ import annotations

import glob
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# All paths must be relative to the directory containing setup.py
# (pip enforces this for editable installs).
EXT_SRC_REL = os.path.join("pn4d", "ops", "_ext_src")

_sources = sorted(
    glob.glob(os.path.join(EXT_SRC_REL, "src", "*.cpp"))
    + glob.glob(os.path.join(EXT_SRC_REL, "src", "*.cu"))
)
_include_dirs = [
    # absolute include dir is required by nvcc; relative path under
    # "extra_compile_args -I" is the safe form.
    os.path.abspath(os.path.join(EXT_SRC_REL, "include")),
]


def _make_extension() -> CUDAExtension:
    return CUDAExtension(
        name="pn4d._C",
        sources=_sources,
        include_dirs=_include_dirs,
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["-O3"],
        },
    )


setup(
    ext_modules=[_make_extension()],
    cmdclass={"build_ext": BuildExtension},
)
