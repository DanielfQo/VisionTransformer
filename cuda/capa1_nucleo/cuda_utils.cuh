#pragma once

#include <cuda_runtime_api.h>
#include <math_constants.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace vit {

// ---- utilidades compartidas por todos los kernels ----

inline void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

inline void require_device_pointer(const float* pointer, const char* name) {
    if (pointer == nullptr) {
        throw std::invalid_argument(std::string(name) + " no puede ser nullptr");
    }
}

constexpr int kBlockSize = 256;

// block_reduce_* asume blockDim.x multiplo de 32 (warps completos) y <= 1024.
static_assert(kBlockSize % 32 == 0 && kBlockSize <= 1024,
              "kBlockSize debe ser multiplo de 32 y <= 1024 (requisito de block_reduce_*)");

inline dim3 grid_1d(std::size_t elements, int batch = 1) {
    auto blocks = static_cast<unsigned int>((elements + kBlockSize - 1) / kBlockSize);
    return dim3(blocks, 1, static_cast<unsigned int>(batch));
}

// ---- Parte 2: Reducciones optimizadas con Warp Shuffle (__shfl_down_sync) ----
// requisito: todos los hilos del bloque llaman a block_reduce_* (sin divergencia)
// y blockDim.x es multiplo de 32.

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_max(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    val = warp_reduce_max(val);

    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    val = (threadIdx.x < (blockDim.x / 32)) ? shared[lane] : -CUDART_INF_F;
    if (warp_id == 0) {
        val = warp_reduce_max(val);
    }
    __syncthreads();  // permite llamadas consecutivas sin carrera sobre shared[]
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    val = (threadIdx.x < (blockDim.x / 32)) ? shared[lane] : 0.0f;
    if (warp_id == 0) {
        val = warp_reduce_sum(val);
    }
    __syncthreads();  // permite llamadas consecutivas sin carrera sobre shared[]
    return val;
}

}  // namespace vit
