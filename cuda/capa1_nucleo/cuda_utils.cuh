#pragma once

#include <cuda_runtime_api.h>

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

inline dim3 grid_1d(std::size_t elements, int batch = 1) {
    auto blocks = static_cast<unsigned int>((elements + kBlockSize - 1) / kBlockSize);
    return dim3(blocks, 1, static_cast<unsigned int>(batch));
}

}  // namespace vit
