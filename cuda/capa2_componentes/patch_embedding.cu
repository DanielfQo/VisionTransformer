#include "patch_embedding.cuh"

#include <stdexcept>
#include <string>

namespace vit {
namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ std::size_t image_index(
    int b,
    int c,
    int row,
    int col,
    int channels,
    int height,
    int width) {
    return ((static_cast<std::size_t>(b) * channels + c) * height + row) * width + col;
}

__device__ __forceinline__ std::size_t token_index(
    int b,
    int token,
    int d,
    int seq_len,
    int d_model) {
    return (static_cast<std::size_t>(b) * seq_len + token) * d_model + d;
}

__global__ void patch_embedding_kernel(
    const float* image,
    const float* weight,
    const float* bias,
    float* patch_tokens,
    int channels,
    int height,
    int width,
    int patch_size,
    int patches_w,
    int n_patches,
    int d_model) {
    const std::size_t total = static_cast<std::size_t>(n_patches) * d_model;
    const std::size_t linear =
        (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (linear >= total) {
        return;
    }

    const int b = blockIdx.z;
    const int d = static_cast<int>(linear % d_model);
    const int patch = static_cast<int>((linear / d_model) % n_patches);
    const int patch_row = patch / patches_w;
    const int patch_col = patch % patches_w;

    float sum = bias == nullptr ? 0.0F : bias[d];
    int flat_patch = 0;
    for (int c = 0; c < channels; ++c) {
        for (int i = 0; i < patch_size; ++i) {
            for (int j = 0; j < patch_size; ++j) {
                const int row = patch_row * patch_size + i;
                const int col = patch_col * patch_size + j;
                const float pixel =
                    image[image_index(b, c, row, col, channels, height, width)];
                sum += pixel * weight[static_cast<std::size_t>(flat_patch) * d_model + d];
                ++flat_patch;
            }
        }
    }

    patch_tokens[token_index(b, patch, d, n_patches, d_model)] = sum;
}

__global__ void add_cls_token_kernel(
    const float* patch_tokens,
    const float* cls_token,
    float* tokens_with_cls,
    int n_patches,
    int d_model) {
    const int seq_len = n_patches + 1;
    const std::size_t total = static_cast<std::size_t>(seq_len) * d_model;
    const std::size_t linear =
        (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (linear >= total) {
        return;
    }

    const int b = blockIdx.z;
    const int d = static_cast<int>(linear % d_model);
    const int token = static_cast<int>((linear / d_model) % seq_len);

    if (token == 0) {
        tokens_with_cls[token_index(b, token, d, seq_len, d_model)] =
            cls_token == nullptr ? 0.0F : cls_token[d];
        return;
    }

    tokens_with_cls[token_index(b, token, d, seq_len, d_model)] =
        patch_tokens[token_index(b, token - 1, d, n_patches, d_model)];
}

__global__ void add_position_embedding_kernel(
    const float* tokens,
    const float* position_embedding,
    float* output,
    int seq_len,
    int d_model) {
    const std::size_t total = static_cast<std::size_t>(seq_len) * d_model;
    const std::size_t linear =
        (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (linear >= total) {
        return;
    }

    const int b = blockIdx.z;
    const int d = static_cast<int>(linear % d_model);
    const int token = static_cast<int>((linear / d_model) % seq_len);
    const std::size_t out_index = token_index(b, token, d, seq_len, d_model);
    output[out_index] =
        tokens[out_index] + position_embedding[static_cast<std::size_t>(token) * d_model + d];
}

__global__ void patch_embedding_pipeline_kernel(
    const float* image,
    const float* weight,
    const float* bias,
    const float* cls_token,
    const float* position_embedding,
    float* output,
    int channels,
    int height,
    int width,
    int patch_size,
    int patches_w,
    int n_patches,
    int d_model) {
    const int seq_len = n_patches + 1;
    const std::size_t total = static_cast<std::size_t>(seq_len) * d_model;
    const std::size_t linear =
        (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (linear >= total) {
        return;
    }

    const int b = blockIdx.z;
    const int d = static_cast<int>(linear % d_model);
    const int token = static_cast<int>((linear / d_model) % seq_len);

    float value = 0.0F;
    if (token == 0) {
        value = cls_token == nullptr ? 0.0F : cls_token[d];
    } else {
        const int patch = token - 1;
        const int patch_row = patch / patches_w;
        const int patch_col = patch % patches_w;
        value = bias == nullptr ? 0.0F : bias[d];

        int flat_patch = 0;
        for (int c = 0; c < channels; ++c) {
            for (int i = 0; i < patch_size; ++i) {
                for (int j = 0; j < patch_size; ++j) {
                    const int row = patch_row * patch_size + i;
                    const int col = patch_col * patch_size + j;
                    const float pixel =
                        image[image_index(b, c, row, col, channels, height, width)];
                    value += pixel *
                             weight[static_cast<std::size_t>(flat_patch) * d_model + d];
                    ++flat_patch;
                }
            }
        }
    }

    value += position_embedding[static_cast<std::size_t>(token) * d_model + d];
    output[token_index(b, token, d, seq_len, d_model)] = value;
}

void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

dim3 grid_for(std::size_t elements, int batch) {
    const auto blocks =
        static_cast<unsigned int>((elements + kThreads - 1) / kThreads);
    return dim3(blocks, 1, static_cast<unsigned int>(batch));
}

void require_device_pointer(const float* pointer, const char* name) {
    if (pointer == nullptr) {
        throw std::invalid_argument(std::string(name) + " no puede ser nullptr");
    }
}

}

int PatchEmbeddingShape::patches_h() const {
    return height / patch_size;
}

int PatchEmbeddingShape::patches_w() const {
    return width / patch_size;
}

int PatchEmbeddingShape::n_patches() const {
    return patches_h() * patches_w();
}

int PatchEmbeddingShape::patch_dim() const {
    return in_channels * patch_size * patch_size;
}

int PatchEmbeddingShape::seq_len() const {
    return n_patches() + 1;
}

std::size_t PatchEmbeddingShape::image_elements() const {
    return static_cast<std::size_t>(batch) * in_channels * height * width;
}

std::size_t PatchEmbeddingShape::patch_token_elements() const {
    return static_cast<std::size_t>(batch) * n_patches() * d_model;
}

std::size_t PatchEmbeddingShape::output_elements() const {
    return static_cast<std::size_t>(batch) * seq_len() * d_model;
}

std::size_t PatchEmbeddingShape::weight_elements() const {
    return static_cast<std::size_t>(patch_dim()) * d_model;
}

std::size_t PatchEmbeddingShape::position_elements() const {
    return static_cast<std::size_t>(seq_len()) * d_model;
}

void PatchEmbeddingShape::validate() const {
    if (batch <= 0 || in_channels <= 0 || height <= 0 || width <= 0 ||
        patch_size <= 0 || d_model <= 0) {
        throw std::invalid_argument(
            "Las dimensiones de patch embedding deben ser positivas");
    }
    if (height % patch_size != 0 || width % patch_size != 0) {
        throw std::invalid_argument(
            "height y width deben ser divisibles entre patch_size");
    }
}

