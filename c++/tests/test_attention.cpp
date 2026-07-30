#include "../capa2_componentes/attention.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

bool close(float a, float b, float tolerance = 1e-5F) {
    return std::fabs(a - b) <= tolerance;
}

}  // namespace

int main() {
    const vit::AttentionShape shape{1, 2, 4, 2};
    const std::vector<float> input{
        1.0F, 0.0F, 1.0F, 0.0F,
        0.0F, 1.0F, 0.0F, 1.0F};

    std::vector<float> weights;
    const auto output =
        vit::multi_head_self_attention_cpu(input, shape, nullptr, &weights);

    assert(output.size() == input.size());
    assert(weights.size() == shape.weight_elements());
    for (int h = 0; h < shape.heads; ++h) {
        for (int q = 0; q < shape.tokens; ++q) {
            const std::size_t row = (h * shape.tokens + q) * shape.tokens;
            assert(close(weights[row] + weights[row + 1], 1.0F));
        }
    }

    const std::vector<std::uint8_t> identity_mask{
        1, 0, 0, 1,
        1, 0, 0, 1};
    const auto masked =
        vit::multi_head_self_attention_cpu(input, shape, &identity_mask);
    for (std::size_t i = 0; i < input.size(); ++i) {
        assert(close(masked[i], input[i]));
    }

    Tensor tokens({1, 2, 4}, 0.0F);
    std::copy(input.begin(), input.end(), tokens.data());
    MultiHeadSelfAttention attention(4, 2);
    const Tensor tensor_output = attention.forward(tokens);
    assert(tensor_output.shape() == tokens.shape());
    for (std::size_t i = 0; i < output.size(); ++i) {
        assert(close(tensor_output.data()[i], output[i]));
    }

    std::cout << "attention CPU: OK\n";
    return 0;
}
