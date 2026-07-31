#pragma once

#include "../../c++/capa2_componentes/encoder.hpp"

#include <cuda_runtime_api.h>

using namespace std;

namespace vit {

// un bloque por fila; normaliza en paralelo
void layer_norm_cuda(const float* d_input, const float* d_gamma,
                     const float* d_beta, float* d_output,
                     const LayerNormShape& shape, float eps,
                     cudaStream_t stream = nullptr);

// un hilo por elemento de salida
void linear_cuda(const float* d_input, const float* d_weight,
                 const float* d_bias, float* d_output,
                 const LinearShape& shape, cudaStream_t stream = nullptr);

// un hilo por elemento
void gelu_cuda(const float* d_input, float* d_output, size_t n,
               cudaStream_t stream = nullptr);

}  // namespace vit
