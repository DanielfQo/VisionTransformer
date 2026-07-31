#include "softmax.cuh"

#include <stdexcept>

namespace vit {
namespace {

// un bloque por fila; reduccion en shared memory
__global__ void softmax_kernel(const float* input, float* output,
                               int cols) {
    int row = blockIdx.x;
    const float* x = input + static_cast<std::size_t>(row) * cols;
    float* y = output + static_cast<std::size_t>(row) * cols;

    extern __shared__ float shared[];

    // paso 1: maximo para estabilidad numerica
    float local_max = -CUDART_INF_F;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        local_max = fmaxf(local_max, x[i]);
    }
    shared[threadIdx.x] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x],
                                        shared[threadIdx.x + s]);
        }
        __syncthreads();
    }
    float max_val = shared[0];
    __syncthreads();

    // paso 2: exp y suma
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float e = expf(x[i] - max_val);
        y[i] = e;
        local_sum += e;
    }
    shared[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }
    float total = shared[0];

    // paso 3: normalizar
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

    std::size_t shared_bytes = kBlockSize * sizeof(float);
    softmax_kernel<<<rows, kBlockSize, shared_bytes, stream>>>(
        d_input, d_output, cols);
    check_cuda(cudaGetLastError(), "No se pudo lanzar softmax_kernel");
}

}  // namespace vit
