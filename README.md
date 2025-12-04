# graphgp-cuda

This is the custom CUDA extension for [GraphGP](https://github.com/stanford-ism/graphgp). It is not intended to be used as a standalone package.

The graph construction uses two GPU-friendly tree algorithms [[2](https://arxiv.org/abs/2211.00120), [3](https://arxiv.org/abs/2210.12859)], which were originally implemented in the [cudaKDTree](https://github.com/ingowald/cudaKDTree) library.

For in-place sorts, we use a modified version of the Apache 2.0-licensed [cubit](https://github.com/ingowald/cudaBitonic) bitonic sorting code, which can be found in "graphgp_cuda/sort.h".