#include "encoder.cuh"

#include <stdexcept>
#include <string>

using namespace std;

namespace vit {
namespace {

const int kThreads = 256;

// una fila por bloque; reduccion en memoria compartida para media y varianza
__global__ void layer_norm_kernel(const float* input, const float* gamma,
                                  const float* beta, float* output,
                                  int dim, float eps) {
    int row = blockIdx.x;
    const float* x = input + (size_t)row * dim;
    float* y = output + (size_t)row * dim;

    extern __shared__ float red[];

    // media
    float local = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        local += x[i];
    }
    red[threadIdx.x] = local;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (threadIdx.x < s) {
            red[threadIdx.x] += red[threadIdx.x + s];
        }
        __syncthreads();
    }
    float mean = red[0] / (float)dim;
    __syncthreads();

    // varianza
    float lv = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float d = x[i] - mean;
        lv += d * d;
    }
    red[threadIdx.x] = lv;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (threadIdx.x < s) {
            red[threadIdx.x] += red[threadIdx.x + s];
        }
        __syncthreads();
    }
    float inv = rsqrtf(red[0] / (float)dim + eps);

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

    const float* x = input + (size_t)r * in_features;
    float acc = bias != nullptr ? bias[o] : 0.0f;
    for (int i = 0; i < in_features; ++i) {
        acc += x[i] * weight[(size_t)i * out_features + o];
    }
    output[(size_t)r * out_features + o] = acc;
}


__global__ void gelu_kernel(const float* input, float* output, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    float x = input[i];
    float k = 0.7978845608028654f;  // sqrt(2/pi)
    float c = 0.044715f;
    output[i] = 0.5f * x * (1.0f + tanhf(k * (x + c * x * x * x)));
}


void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw runtime_error(string(operation) + ": " + cudaGetErrorString(error));
    }
}

}  // namespace


void layer_norm_cuda(const float* d_input, const float* d_gamma,
                     const float* d_beta, float* d_output,
                     const LayerNormShape& shape, float eps,
                     cudaStream_t stream) {
    shape.validate();
    if (d_input == nullptr || d_output == nullptr) {
        throw invalid_argument("layer_norm_cuda: punteros nulos");
    }

    size_t shared_bytes = kThreads * sizeof(float);
    layer_norm_kernel<<<shape.rows, kThreads, shared_bytes, stream>>>(
        d_input, d_gamma, d_beta, d_output, shape.dim, eps);
    check_cuda(cudaGetLastError(), "No se pudo lanzar layer_norm_kernel");
}


void linear_cuda(const float* d_input, const float* d_weight,
                 const float* d_bias, float* d_output,
                 const LinearShape& shape, cudaStream_t stream) {
    shape.validate();
    if (d_input == nullptr || d_output == nullptr || d_weight == nullptr) {
        throw invalid_argument("linear_cuda: punteros nulos");
    }

    dim3 grid((shape.out_features + kThreads - 1) / kThreads, shape.rows);
    linear_kernel<<<grid, kThreads, 0, stream>>>(
        d_input, d_weight, d_bias, d_output, shape.in_features,
        shape.out_features);
    check_cuda(cudaGetLastError(), "No se pudo lanzar linear_kernel");
}


void gelu_cuda(const float* d_input, float* d_output, size_t n,
               cudaStream_t stream) {
    if (d_input == nullptr || d_output == nullptr) {
        throw invalid_argument("gelu_cuda: punteros nulos");
    }

    size_t blocks = (n + kThreads - 1) / kThreads;
    gelu_kernel<<<blocks, kThreads, 0, stream>>>(d_input, d_output, n);
    check_cuda(cudaGetLastError(), "No se pudo lanzar gelu_kernel");
}

}  // namespace vit
