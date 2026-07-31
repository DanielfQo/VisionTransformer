#include "softmax.cuh"

#include <stdexcept>

namespace vit {
namespace {

// un bloque por fila; reduccion rapida usando warp shuffle
__global__ void softmax_kernel(const float* input, float* output, int cols) {
    int row = blockIdx.x;
    const float* x = input + static_cast<std::size_t>(row) * cols;
    float* y = output + static_cast<std::size_t>(row) * cols;

    // paso 1: maximo con warp shuffle
    float local_max = -CUDART_INF_F;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        local_max = fmaxf(local_max, x[i]);
    }
    float max_val = block_reduce_max(local_max);
    __shared__ float s_max;
    if (threadIdx.x == 0) {
        s_max = max_val;
    }
    __syncthreads();
    max_val = s_max;

    // paso 2: exp y suma con warp shuffle
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float e = expf(x[i] - max_val);
        y[i] = e;
        local_sum += e;
    }
    float total = block_reduce_sum(local_sum);
    __shared__ float s_total;
    if (threadIdx.x == 0) {
        s_total = total;
    }
    __syncthreads();
    total = s_total;

    // paso 3: normalizacion
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        y[i] /= total;
    }
}

__global__ void softmax_backward_kernel(const float* s_out, const float* g_out,
                                        float* g_in, int cols) {
    int row = blockIdx.x;
    const float* s = s_out + static_cast<std::size_t>(row) * cols;
    const float* dy = g_out + static_cast<std::size_t>(row) * cols;
    float* dx = g_in + static_cast<std::size_t>(row) * cols;

    float local_dot = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        local_dot += dy[i] * s[i];
    }
    float total_dot = block_reduce_sum(local_dot);
    __shared__ float s_dot;
    if (threadIdx.x == 0) {
        s_dot = total_dot;
    }
    __syncthreads();
    float dot = s_dot;

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        dx[i] = s[i] * (dy[i] - dot);
    }
}

}  // namespace

void softmax_cuda(const float* d_input, float* d_output,
                  int rows, int cols,
                  cudaStream_t stream) {
    if (d_input == nullptr || d_output == nullptr) {
        throw std::invalid_argument("softmax_cuda: punteros nulos");
    }
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument("softmax_cuda: dimensiones deben ser positivas");
    }

    softmax_kernel<<<rows, kBlockSize, 0, stream>>>(d_input, d_output, cols);
    check_cuda(cudaGetLastError(), "No se pudo lanzar softmax_kernel");
}

void softmax_backward_cuda(const float* d_softmax_out, const float* d_grad_out,
                           float* d_grad_in, int rows, int cols,
                           cudaStream_t stream) {
    if (d_softmax_out == nullptr || d_grad_out == nullptr || d_grad_in == nullptr) {
        throw std::invalid_argument("softmax_backward_cuda: punteros nulos");
    }
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument("softmax_backward_cuda: dimensiones deben ser positivas");
    }

    softmax_backward_kernel<<<rows, kBlockSize, 0, stream>>>(
        d_softmax_out, d_grad_out, d_grad_in, cols);
    check_cuda(cudaGetLastError(), "No se pudo lanzar softmax_backward_kernel");
}

}  // namespace vit
