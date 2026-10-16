# graphgp-cuda

This is the custom CUDA extension for [GraphGP](https://github.com/stanford-ism/graphgp). It is not intended to be used as a standalone package.

The graph construction uses two GPU-friendly tree algorithms [[2](https://arxiv.org/abs/2211.00120), [3](https://arxiv.org/abs/2210.12859)], which were originally implemented in the [cudaKDTree](https://github.com/ingowald/cudaKDTree) library.

For in-place sorts, we use a modified version of the Apache 2.0-licensed [cubit](https://github.com/ingowald/cudaBitonic) bitonic sorting code, which can be found in "graphgp_cuda/sort.h".
## Building

`pip install .` builds the extension with the CUDA 13 toolkit from pip (`nvidia-cuda-nvcc`, pulled in as a build requirement), so no system CUDA installation is needed. To use a different compiler, set `CUDACXX` or pass `-C cmake.define.CMAKE_CUDA_COMPILER=...`.

By default the library is compiled for compute capabilities 80, 89, 90 and 120 (A100, RTX 40xx, H100, RTX 50xx), with PTX so newer GPUs can JIT compile it. For a faster local build, target only your GPU and compile a reduced set of kernels:

```bash
pip install "jax[cuda13]" scikit-build-core "cmake>=3.24" ninja pytest   # once, the build tools and toolkit
CUDAARCHS=native pip install --no-build-isolation . -C cmake.define.GRAPHGP_CUDA_DEV_BUILD=ON
```

The development build only supports `k <= 16` and `n_dim` in {2, 3}.
