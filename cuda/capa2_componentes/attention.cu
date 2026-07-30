#include "attention.cuh"

#include <cmath>
#include <stdexcept>
#include <string>

namespace vit {
namespace {

constexpr int kThreads = 256;

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
        const float weight = denominator > 0.0F ? scores[k] / denominator : 0.0F;
        scores[k] = weight;
        if (weights != nullptr) {
            weights[row * tokens + k] = weight;
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

void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

}

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
        static_cast<std::size_t>(shape.tokens + kThreads) * sizeof(float);
    attention_kernel<<<grid, kThreads, shared_bytes, stream>>>(
        d_input,
        d_output,
        d_mask,
        d_attention_weights,
        shape.tokens,
        shape.d_model,
        shape.heads);
    check_cuda(cudaGetLastError(), "No se pudo lanzar attention_kernel");
}

}
