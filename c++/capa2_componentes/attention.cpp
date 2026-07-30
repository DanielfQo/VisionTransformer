#include "attention.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace vit {

int AttentionShape::head_dim() const {
    return d_model / heads;
}

std::size_t AttentionShape::token_elements() const {
    return static_cast<std::size_t>(batch) * tokens * d_model;
}

std::size_t AttentionShape::weight_elements() const {
    return static_cast<std::size_t>(batch) * heads * tokens * tokens;
}

void AttentionShape::validate() const {
    if (batch <= 0 || tokens <= 0 || d_model <= 0 || heads <= 0) {
        throw std::invalid_argument("Las dimensiones de attention deben ser positivas");
    }
    if (d_model % heads != 0) {
        throw std::invalid_argument("d_model debe ser divisible entre heads");
    }
}

std::vector<float> multi_head_self_attention_cpu(
    const std::vector<float>& input,
    const AttentionShape& shape,
    const std::vector<std::uint8_t>* mask,
    std::vector<float>* attention_weights) {
    shape.validate();
    if (input.size() != shape.token_elements()) {
        throw std::invalid_argument("El tamano de input no coincide con AttentionShape");
    }
    if (mask != nullptr && mask->size() != shape.weight_elements()) {
        throw std::invalid_argument("La mascara debe tener forma [batch, heads, tokens, tokens]");
    }

    std::vector<float> output(shape.token_elements(), 0.0F);
    if (attention_weights != nullptr) {
        attention_weights->assign(shape.weight_elements(), 0.0F);
    }

    const int dim = shape.head_dim();
    const float scale = 1.0F / std::sqrt(static_cast<float>(dim));
    std::vector<float> scores(shape.tokens);

    for (int b = 0; b < shape.batch; ++b) {
        for (int h = 0; h < shape.heads; ++h) {
            const int feature = h * dim;
            for (int q = 0; q < shape.tokens; ++q) {
                float maximum = -std::numeric_limits<float>::infinity();
                bool any_allowed = false;

                for (int k = 0; k < shape.tokens; ++k) {
                    const std::size_t weight_index =
                        ((static_cast<std::size_t>(b) * shape.heads + h) *
                             shape.tokens +
                         q) *
                            shape.tokens +
                        k;
                    const bool allowed = mask == nullptr || (*mask)[weight_index] != 0;
                    any_allowed = any_allowed || allowed;

                    float dot = 0.0F;
                    for (int d = 0; d < dim; ++d) {
                        const std::size_t q_index =
                            (static_cast<std::size_t>(b) * shape.tokens + q) *
                                shape.d_model +
                            feature + d;
                        const std::size_t k_index =
                            (static_cast<std::size_t>(b) * shape.tokens + k) *
                                shape.d_model +
                            feature + d;
                        dot += input[q_index] * input[k_index];
                    }
                    scores[k] = allowed ? dot * scale
                                        : -std::numeric_limits<float>::infinity();
                    maximum = std::max(maximum, scores[k]);
                }

                if (!any_allowed) {
                    continue;
                }

                float denominator = 0.0F;
                for (int k = 0; k < shape.tokens; ++k) {
                    scores[k] = std::exp(scores[k] - maximum);
                    denominator += scores[k];
                }

                for (int k = 0; k < shape.tokens; ++k) {
                    const float weight = scores[k] / denominator;
                    if (attention_weights != nullptr) {
                        const std::size_t weight_index =
                            ((static_cast<std::size_t>(b) * shape.heads + h) *
                                 shape.tokens +
                             q) *
                                shape.tokens +
                            k;
                        (*attention_weights)[weight_index] = weight;
                    }
                    for (int d = 0; d < dim; ++d) {
                        const std::size_t output_index =
                            (static_cast<std::size_t>(b) * shape.tokens + q) *
                                shape.d_model +
                            feature + d;
                        const std::size_t value_index =
                            (static_cast<std::size_t>(b) * shape.tokens + k) *
                                shape.d_model +
                            feature + d;
                        output[output_index] += weight * input[value_index];
                    }
                }
            }
        }
    }
    return output;
}

}

MultiHeadSelfAttention::MultiHeadSelfAttention(
    std::size_t d_model,
    std::size_t n_heads)
    : d_model_(d_model), n_heads_(n_heads) {
    if (d_model_ == 0 || n_heads_ == 0) {
        throw std::invalid_argument(
            "MultiHeadSelfAttention dimensions must be greater than zero");
    }
    if (d_model_ % n_heads_ != 0) {
        throw std::invalid_argument("d_model must be divisible by n_heads");
    }
}

Tensor MultiHeadSelfAttention::forward(const Tensor& x) {
    if (x.ndim() != 3) {
        throw std::invalid_argument(
            "MultiHeadSelfAttention expects shape [batch, tokens, d_model]");
    }
    if (x.dim(2) != d_model_) {
        throw std::invalid_argument(
            "MultiHeadSelfAttention received an unexpected d_model");
    }

    const vit::AttentionShape shape{
        static_cast<int>(x.dim(0)),
        static_cast<int>(x.dim(1)),
        static_cast<int>(x.dim(2)),
        static_cast<int>(n_heads_)};
    const std::vector<float> input(x.data(), x.data() + x.numel());
    const std::vector<float> values =
        vit::multi_head_self_attention_cpu(input, shape);
    Tensor output({x.dim(0), x.dim(1), x.dim(2)}, 0.0F);
    std::copy(values.begin(), values.end(), output.data());
    return output;
}

std::size_t MultiHeadSelfAttention::d_model() const {
    return d_model_;
}

std::size_t MultiHeadSelfAttention::n_heads() const {
    return n_heads_;
}
