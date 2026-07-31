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


// multiplicacion matricial optimizada con tiling en memoria compartida (16x16)
constexpr int kTileSize = 16;

__global__ void linear_kernel(const float* input, const float* weight,
                              const float* bias, float* output,
                              int in_features, int out_features, int rows) {
    __shared__ float s_in[kTileSize][kTileSize];
    __shared__ float s_w[kTileSize][kTileSize];

    int col = blockIdx.x * kTileSize + threadIdx.x;
    int row = blockIdx.y * kTileSize + threadIdx.y;

    float acc = 0.0f;
    int num_tiles = (in_features + kTileSize - 1) / kTileSize;

    for (int t = 0; t < num_tiles; ++t) {
        int in_col = t * kTileSize + threadIdx.x;
        if (row < rows && in_col < in_features) {
            s_in[threadIdx.y][threadIdx.x] = input[static_cast<std::size_t>(row) * in_features + in_col];
        } else {
            s_in[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int in_row = t * kTileSize + threadIdx.y;
        if (in_row < in_features && col < out_features) {
            s_w[threadIdx.y][threadIdx.x] = weight[static_cast<std::size_t>(in_row) * out_features + col];
        } else {
            s_w[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < kTileSize; ++k) {
            acc += s_in[threadIdx.y][k] * s_w[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < rows && col < out_features) {
        if (bias != nullptr) {
            acc += bias[col];
        }
        output[static_cast<std::size_t>(row) * out_features + col] = acc;
    }
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

    dim3 block(kTileSize, kTileSize);
    dim3 grid((shape.out_features + kTileSize - 1) / kTileSize,
              (shape.rows + kTileSize - 1) / kTileSize);
    linear_kernel<<<grid, block, 0, stream>>>(
        d_input, d_weight, d_bias, d_output, shape.in_features,
        shape.out_features, shape.rows);
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

// ---- CUDA backward kernels ----

namespace {

__global__ void layer_norm_backward_kernel(const float* input, const float* gamma,
                                           const float* grad_out, float* grad_in,
                                           float* grad_gamma, float* grad_beta,
                                           int dim, float eps) {
    int row = blockIdx.x;
    const float* x = input + static_cast<std::size_t>(row) * dim;
    const float* dy = grad_out + static_cast<std::size_t>(row) * dim;
    float* dx = grad_in != nullptr ? grad_in + static_cast<std::size_t>(row) * dim : nullptr;

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

    float local_dhat = 0.0f;
    float local_dhat_xhat = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float g = gamma != nullptr ? gamma[i] : 1.0f;
        float xhat = (x[i] - mean) * inv;
        float dhat = dy[i] * g;
        local_dhat += dhat;
        local_dhat_xhat += dhat * xhat;

        if (grad_gamma != nullptr) {
            atomicAdd(&grad_gamma[i], dy[i] * xhat);
        }
        if (grad_beta != nullptr) {
            atomicAdd(&grad_beta[i], dy[i]);
        }
    }

    float total_dhat = block_reduce_sum(local_dhat);
    __syncthreads();
    float total_dhat_xhat = block_reduce_sum(local_dhat_xhat);
    __shared__ float s_dhat, s_dhat_xhat;
    if (threadIdx.x == 0) {
        s_dhat = total_dhat;
        s_dhat_xhat = total_dhat_xhat;
    }
    __syncthreads();

    if (dx != nullptr) {
        float inv_N = 1.0f / static_cast<float>(dim);
        float s1 = s_dhat;
        float s2 = s_dhat_xhat;
        for (int i = threadIdx.x; i < dim; i += blockDim.x) {
            float xhat = (x[i] - mean) * inv;
            float g = gamma != nullptr ? gamma[i] : 1.0f;
            float dhat = dy[i] * g;
            dx[i] = inv * (dhat - inv_N * (s1 + xhat * s2));
        }
    }
}

__global__ void linear_backward_input_kernel(const float* weight, const float* grad_out,
                                             float* grad_in, int in_features, int out_features) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y;
    if (i >= in_features) return;

    const float* dy = grad_out + static_cast<std::size_t>(r) * out_features;
    float acc = 0.0f;
    for (int o = 0; o < out_features; ++o) {
        acc += dy[o] * weight[static_cast<std::size_t>(i) * out_features + o];
    }
    grad_in[static_cast<std::size_t>(r) * in_features + i] = acc;
}

__global__ void linear_backward_params_kernel(const float* input, const float* grad_out,
                                              float* grad_weight, float* grad_bias,
                                              int rows, int in_features, int out_features) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y;
    if (o >= out_features || i >= in_features) return;

    float acc_w = 0.0f;
    for (int r = 0; r < rows; ++r) {
        float x = input[static_cast<std::size_t>(r) * in_features + i];
        float dy = grad_out[static_cast<std::size_t>(r) * out_features + o];
        acc_w += x * dy;
    }
    if (grad_weight != nullptr) {
        atomicAdd(&grad_weight[static_cast<std::size_t>(i) * out_features + o], acc_w);
    }

    if (i == 0 && grad_bias != nullptr) {
        float acc_b = 0.0f;
        for (int r = 0; r < rows; ++r) {
            acc_b += grad_out[static_cast<std::size_t>(r) * out_features + o];
        }
        atomicAdd(&grad_bias[o], acc_b);
    }
}

__global__ void gelu_backward_kernel(const float* input, const float* grad_out,
                                    float* grad_in, std::size_t n) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float x = input[i];
    float dy = grad_out[i];
    float k = 0.7978845608028654f;  // sqrt(2/pi)
    float c = 0.044715f;
    float u = k * (x + c * x * x * x);
    float t = tanhf(u);
    float du_dx = k * (1.0f + 3.0f * c * x * x);
    float dx = 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * du_dx;
    grad_in[i] = dy * dx;
}

}  // namespace

void layer_norm_backward_cuda(const float* d_input, const float* d_gamma,
                              const float* d_grad_out, float* d_grad_in,
                              float* d_grad_gamma, float* d_grad_beta,
                              const LayerNormShape& shape, float eps,
                              cudaStream_t stream) {
    shape.validate();
    if (d_input == nullptr || d_grad_out == nullptr) {
        throw std::invalid_argument("layer_norm_backward_cuda: punteros nulos");
    }

    layer_norm_backward_kernel<<<shape.rows, kBlockSize, 0, stream>>>(
        d_input, d_gamma, d_grad_out, d_grad_in, d_grad_gamma, d_grad_beta, shape.dim, eps);
    check_cuda(cudaGetLastError(), "No se pudo lanzar layer_norm_backward_kernel");
}

void linear_backward_cuda(const float* d_input, const float* d_weight,
                          const float* d_grad_out, float* d_grad_in,
                          float* d_grad_weight, float* d_grad_bias,
                          const LinearShape& shape,
                          cudaStream_t stream) {
    shape.validate();
    if (d_input == nullptr || d_weight == nullptr || d_grad_out == nullptr) {
        throw std::invalid_argument("linear_backward_cuda: punteros nulos");
    }

    if (d_grad_in != nullptr) {
        dim3 grid_in((shape.in_features + kBlockSize - 1) / kBlockSize, shape.rows);
        linear_backward_input_kernel<<<grid_in, kBlockSize, 0, stream>>>(
            d_weight, d_grad_out, d_grad_in, shape.in_features, shape.out_features);
        check_cuda(cudaGetLastError(), "No se pudo lanzar linear_backward_input_kernel");
    }

    if (d_grad_weight != nullptr || d_grad_bias != nullptr) {
        dim3 grid_p((shape.out_features + kBlockSize - 1) / kBlockSize, shape.in_features);
        linear_backward_params_kernel<<<grid_p, kBlockSize, 0, stream>>>(
            d_input, d_grad_out, d_grad_weight, d_grad_bias, shape.rows, shape.in_features, shape.out_features);
        check_cuda(cudaGetLastError(), "No se pudo lanzar linear_backward_params_kernel");
    }
}

void gelu_backward_cuda(const float* d_input, const float* d_grad_out,
                        float* d_grad_in, std::size_t n,
                        cudaStream_t stream) {
    if (d_input == nullptr || d_grad_out == nullptr || d_grad_in == nullptr) {
        throw std::invalid_argument("gelu_backward_cuda: punteros nulos");
    }

    std::size_t blocks = (n + kBlockSize - 1) / kBlockSize;
    gelu_backward_kernel<<<blocks, kBlockSize, 0, stream>>>(d_input, d_grad_out, d_grad_in, n);
    check_cuda(cudaGetLastError(), "No se pudo lanzar gelu_backward_kernel");
}

}  // namespace vit
