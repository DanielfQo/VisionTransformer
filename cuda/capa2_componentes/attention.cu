#include "attention.cuh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace vit {

// ---- AttentionShape methods ----

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

// ---- CPU reference ----

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
                    const float w = scores[k] / denominator;
                    if (attention_weights != nullptr) {
                        const std::size_t weight_index =
                            ((static_cast<std::size_t>(b) * shape.heads + h) *
                                 shape.tokens +
                             q) *
                                shape.tokens +
                            k;
                        (*attention_weights)[weight_index] = w;
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
                        output[output_index] += w * input[value_index];
                    }
                }
            }
        }
    }
    return output;
}

// ---- CUDA kernel ----

namespace {

__global__ void attention_kernel(
    const float* input,
    float* output,
    const std::uint8_t* mask,
    float* weights,
    int tokens,
    int d_model,
    int heads) {
    const int q = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    const int dim = d_model / heads;
    const int feature = h * dim;
    const std::size_t row =
        (static_cast<std::size_t>(b) * heads + h) * tokens + q;

    extern __shared__ float shared[];
    float* scores = shared;
    float* reduction = scores + tokens;

    float local_max = -CUDART_INF_F;
    for (int k = threadIdx.x; k < tokens; k += blockDim.x) {
        const std::size_t weight_index = row * tokens + k;
        if (mask != nullptr && mask[weight_index] == 0) {
            scores[k] = -CUDART_INF_F;
            continue;
        }

        float dot = 0.0F;
        for (int d = 0; d < dim; ++d) {
            const std::size_t q_index =
                (static_cast<std::size_t>(b) * tokens + q) * d_model + feature + d;
            const std::size_t k_index =
                (static_cast<std::size_t>(b) * tokens + k) * d_model + feature + d;
            dot += input[q_index] * input[k_index];
        }
        scores[k] = dot * rsqrtf(static_cast<float>(dim));
        local_max = fmaxf(local_max, scores[k]);
    }

    reduction[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] =
                fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float maximum = reduction[0];

    float local_sum = 0.0F;
    if (maximum != -CUDART_INF_F) {
        for (int k = threadIdx.x; k < tokens; k += blockDim.x) {
            scores[k] = expf(scores[k] - maximum);
            local_sum += scores[k];
        }
    }
    reduction[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float denominator = reduction[0];

    for (int k = threadIdx.x; k < tokens; k += blockDim.x) {
        const float w = denominator > 0.0F ? scores[k] / denominator : 0.0F;
        scores[k] = w;
        if (weights != nullptr) {
            weights[row * tokens + k] = w;
        }
    }
    __syncthreads();

    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float result = 0.0F;
        for (int k = 0; k < tokens; ++k) {
            const std::size_t value_index =
                (static_cast<std::size_t>(b) * tokens + k) * d_model + feature + d;
            result += scores[k] * input[value_index];
        }
        const std::size_t output_index =
            (static_cast<std::size_t>(b) * tokens + q) * d_model + feature + d;
        output[output_index] = result;
    }
}

}  // namespace

void multi_head_self_attention_cuda(
    const float* d_input,
    float* d_output,
    const AttentionShape& shape,
    const std::uint8_t* d_mask,
    float* d_attention_weights,
    cudaStream_t stream) {
    shape.validate();
    if (d_input == nullptr || d_output == nullptr) {
        throw std::invalid_argument("d_input y d_output no pueden ser nullptr");
    }

    const dim3 grid(shape.tokens, shape.heads, shape.batch);
    const std::size_t shared_bytes =
        static_cast<std::size_t>(shape.tokens + kBlockSize) * sizeof(float);
    attention_kernel<<<grid, kBlockSize, shared_bytes, stream>>>(
        d_input,
        d_output,
        d_mask,
        d_attention_weights,
        shape.tokens,
        shape.d_model,
        shape.heads);
    check_cuda(cudaGetLastError(), "No se pudo lanzar attention_kernel");
}

}  // namespace vit
