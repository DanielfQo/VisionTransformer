#pragma once

#include "../capa1_nucleo/module.hpp"

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

}

class MultiHeadSelfAttention : public Module {
public:
    MultiHeadSelfAttention(std::size_t d_model, std::size_t n_heads);

    Tensor forward(const Tensor& x) override;

    std::size_t d_model() const;
    std::size_t n_heads() const;

private:
    std::size_t d_model_;
    std::size_t n_heads_;
};
