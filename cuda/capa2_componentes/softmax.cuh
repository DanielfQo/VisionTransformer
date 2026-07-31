#pragma once

#include "../capa1_nucleo/cuda_utils.cuh"

#include <cstddef>

namespace vit {

// softmax sobre la ultima dimension de un tensor plano [rows, cols]
void softmax_cuda(const float* d_input, float* d_output,
                  int rows, int cols,
                  cudaStream_t stream = nullptr);

}  // namespace vit
