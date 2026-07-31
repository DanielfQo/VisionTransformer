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

// backward de la self-attention sin pesos (Q = K = V = input).
// d_input:    entrada Y usada en el forward            [batch, tokens, d_model]
// d_probs:    pesos de atencion A guardados del forward [batch, heads, tokens, tokens]
// d_grad_out: gradiente respecto a la salida            [batch, tokens, d_model]
// d_grad_in:  gradiente respecto a la entrada (salida)  [batch, tokens, d_model]
// dY = dQ + dK + dV con softmax backward exacto sobre A.
void multi_head_self_attention_backward_cuda(
    const float* d_input,
    const float* d_probs,
    const float* d_grad_out,
    float* d_grad_in,
    const AttentionShape& shape,
    cudaStream_t stream = nullptr);

}  // namespace vit
