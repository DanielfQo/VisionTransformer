#pragma once

#include "../../c++/capa2_componentes/attention.hpp"

#include <cuda_runtime_api.h>

namespace vit {

void multi_head_self_attention_cuda(
    const float* d_input,
    float* d_output,
    const AttentionShape& shape,
    const std::uint8_t* d_mask = nullptr,
    float* d_attention_weights = nullptr,
    cudaStream_t stream = nullptr);

}
