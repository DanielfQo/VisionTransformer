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

}  // namespace vit
