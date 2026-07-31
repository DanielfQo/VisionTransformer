#pragma once

#include "../capa1_nucleo/cuda_utils.cuh"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vit {

struct AttentionShape {
    int batch;
    int tokens;
    int d_model;
    int heads;

    int head_dim() const;
    std::size_t token_elements() const;
    std::size_t weight_elements() const;
    void validate() const;
};

std::vector<float> multi_head_self_attention_cpu(
    const std::vector<float>& input,
    const AttentionShape& shape,
    const std::vector<std::uint8_t>* mask = nullptr,
    std::vector<float>* attention_weights = nullptr);

void multi_head_self_attention_cuda(
    const float* d_input,
    float* d_output,
    const AttentionShape& shape,
    const std::uint8_t* d_mask = nullptr,
    float* d_attention_weights = nullptr,
    cudaStream_t stream = nullptr);

}  // namespace vit
