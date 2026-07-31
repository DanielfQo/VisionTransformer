#include "encoder.cuh"

#include <cmath>
#include <stdexcept>
#include <string>

namespace vit {

// ---- Shape methods ----

std::size_t LayerNormShape::elements() const {
    return static_cast<std::size_t>(rows) * dim;
}

void LayerNormShape::validate() const {
    if (rows <= 0 || dim <= 0) {
        throw std::invalid_argument("LayerNormShape: dimensiones deben ser positivas");
    }
}

std::size_t LinearShape::input_elements() const {
    return static_cast<std::size_t>(rows) * in_features;
}

std::size_t LinearShape::output_elements() const {
    return static_cast<std::size_t>(rows) * out_features;
}

void LinearShape::validate() const {
    if (rows <= 0 || in_features <= 0 || out_features <= 0) {
        throw std::invalid_argument("LinearShape: dimensiones deben ser positivas");
    }
}

// ---- CPU reference implementations ----

void layer_norm_cpu(const float* input, const float* gamma, const float* beta,
                    float* output, const LayerNormShape& shape, float eps) {
    shape.validate();

    for (int r = 0; r < shape.rows; ++r) {
        const float* x = input + static_cast<std::size_t>(r) * shape.dim;
        float* y = output + static_cast<std::size_t>(r) * shape.dim;

        float mean = 0.0f;
        for (int i = 0; i < shape.dim; ++i) {
            mean += x[i];
        }
        mean /= static_cast<float>(shape.dim);

        float var = 0.0f;
        for (int i = 0; i < shape.dim; ++i) {
            float d = x[i] - mean;
            var += d * d;
        }
        var /= static_cast<float>(shape.dim);

        float inv = 1.0f / std::sqrt(var + eps);
        for (int i = 0; i < shape.dim; ++i) {
            y[i] = (x[i] - mean) * inv * gamma[i] + beta[i];
        }
    }
}


void linear_cpu(const float* input, const float* weight, const float* bias,
                float* output, const LinearShape& shape) {
    shape.validate();

    for (int r = 0; r < shape.rows; ++r) {
        const float* x = input + static_cast<std::size_t>(r) * shape.in_features;
        float* y = output + static_cast<std::size_t>(r) * shape.out_features;

        for (int o = 0; o < shape.out_features; ++o) {
            float acc = bias != nullptr ? bias[o] : 0.0f;
            for (int i = 0; i < shape.in_features; ++i) {
                acc += x[i] * weight[static_cast<std::size_t>(i) * shape.out_features + o];
            }
            y[o] = acc;
        }
    }
}


void gelu_cpu(const float* input, float* output, std::size_t n) {
    float k = 0.7978845608028654f;  // sqrt(2/pi)
    float c = 0.044715f;

    for (std::size_t i = 0; i < n; ++i) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + std::tanh(k * (x + c * x * x * x)));
    }
}

// ---- CUDA kernels ----

namespace {

// una fila por bloque; reduccion optimizada con warp shuffle (__shfl_down_sync)
__global__ void layer_norm_kernel(const float* input, const float* gamma,
                                  const float* beta, float* output,
                                  int dim, float eps) {
    int row = blockIdx.x;
    const float* x = input + static_cast<std::size_t>(row) * dim;
    float* y = output + static_cast<std::size_t>(row) * dim;

    // media
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        local_sum += x[i];
    }
    float total_sum = block_reduce_sum(local_sum);
    __shared__ float s_mean;
    if (threadIdx.x == 0) {
        s_mean = total_sum / static_cast<float>(dim);
    }
    __syncthreads();
    float mean = s_mean;

    // varianza
    float local_var = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float d = x[i] - mean;
        local_var += d * d;
    }
    float total_var = block_reduce_sum(local_var);
    __shared__ float s_inv;
    if (threadIdx.x == 0) {
        s_inv = rsqrtf(total_var / static_cast<float>(dim) + eps);
    }
    __syncthreads();
    float inv = s_inv;

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        y[i] = (x[i] - mean) * inv * gamma[i] + beta[i];
    }
}


// un hilo por (fila, feature de salida)
__global__ void linear_kernel(const float* input, const float* weight,
                              const float* bias, float* output,
                              int in_features, int out_features) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y;
    if (o >= out_features) {
        return;
    }

    const float* x = input + static_cast<std::size_t>(r) * in_features;
    float acc = bias != nullptr ? bias[o] : 0.0f;
    for (int i = 0; i < in_features; ++i) {
        acc += x[i] * weight[static_cast<std::size_t>(i) * out_features + o];
    }
    output[static_cast<std::size_t>(r) * out_features + o] = acc;
}


__global__ void gelu_kernel(const float* input, float* output, std::size_t n) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    float x = input[i];
    float k = 0.7978845608028654f;  // sqrt(2/pi)
    float c = 0.044715f;
    output[i] = 0.5f * x * (1.0f + tanhf(k * (x + c * x * x * x)));
}

}  // namespace


void layer_norm_cuda(const float* d_input, const float* d_gamma,
                     const float* d_beta, float* d_output,
                     const LayerNormShape& shape, float eps,
                     cudaStream_t stream) {
    shape.validate();
    if (d_input == nullptr || d_output == nullptr) {
        throw std::invalid_argument("layer_norm_cuda: punteros nulos");
    }

    layer_norm_kernel<<<shape.rows, kBlockSize, 0, stream>>>(
        d_input, d_gamma, d_beta, d_output, shape.dim, eps);
    check_cuda(cudaGetLastError(), "No se pudo lanzar layer_norm_kernel");
}


void linear_cuda(const float* d_input, const float* d_weight,
                 const float* d_bias, float* d_output,
                 const LinearShape& shape, cudaStream_t stream) {
    shape.validate();
    if (d_input == nullptr || d_output == nullptr || d_weight == nullptr) {
        throw std::invalid_argument("linear_cuda: punteros nulos");
    }

    dim3 grid((shape.out_features + kBlockSize - 1) / kBlockSize, shape.rows);
    linear_kernel<<<grid, kBlockSize, 0, stream>>>(
        d_input, d_weight, d_bias, d_output, shape.in_features,
        shape.out_features);
    check_cuda(cudaGetLastError(), "No se pudo lanzar linear_kernel");
}


void gelu_cuda(const float* d_input, float* d_output, std::size_t n,
               cudaStream_t stream) {
    if (d_input == nullptr || d_output == nullptr) {
        throw std::invalid_argument("gelu_cuda: punteros nulos");
    }

    std::size_t blocks = (n + kBlockSize - 1) / kBlockSize;
    gelu_kernel<<<blocks, kBlockSize, 0, stream>>>(d_input, d_output, n);
    check_cuda(cudaGetLastError(), "No se pudo lanzar gelu_kernel");
}

}  // namespace vit
