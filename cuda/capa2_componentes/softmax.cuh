#pragma once

#include "../capa1_nucleo/cuda_utils.cuh"

#include <cstddef>

namespace vit {

// softmax sobre la ultima dimension de un tensor plano [rows, cols]
void softmax_cuda(const float* d_input, float* d_output,
                  int rows, int cols,
                  cudaStream_t stream = nullptr);

// backward softmax: d_grad_in = softmax_out * (d_grad_out - sum(d_grad_out * softmax_out))
void softmax_backward_cuda(const float* d_softmax_out, const float* d_grad_out,
                           float* d_grad_in, int rows, int cols,
                           cudaStream_t stream = nullptr);

}  // namespace vit
