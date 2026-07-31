#include "patch_embedding.cuh"

#include <stdexcept>
#include <string>

namespace vit {
namespace {

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

}  // namespace

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

std::vector<float> patch_embedding_pipeline_cpu(
    const std::vector<float>& image,
    const std::vector<float>& weight,
    const std::vector<float>& bias,
    const std::vector<float>& cls_token,
    const std::vector<float>& position_embedding,
    const PatchEmbeddingShape& shape) {
    shape.validate();
    if (image.size() != shape.image_elements()) {
        throw std::invalid_argument("El tamano de image no coincide con PatchEmbeddingShape");
    }
    if (weight.size() != shape.weight_elements()) {
        throw std::invalid_argument("El tamano de weight debe ser patch_dim * d_model");
    }
    if (!bias.empty() && bias.size() != static_cast<std::size_t>(shape.d_model)) {
        throw std::invalid_argument("bias debe tener forma [d_model]");
    }
    if (!cls_token.empty() &&
        cls_token.size() != static_cast<std::size_t>(shape.d_model)) {
        throw std::invalid_argument("cls_token debe tener forma [d_model]");
    }
    if (position_embedding.size() != shape.position_elements()) {
        throw std::invalid_argument(
            "position_embedding debe tener forma [N + 1, d_model]");
    }

    std::vector<float> output(shape.output_elements(), 0.0F);
    const int patches_w = shape.patches_w();
    const int seq_len = shape.seq_len();

    for (int b = 0; b < shape.batch; ++b) {
        for (int token = 0; token < seq_len; ++token) {
            for (int d = 0; d < shape.d_model; ++d) {
                float value = 0.0F;
                if (token == 0) {
                    value = cls_token.empty() ? 0.0F : cls_token[d];
                } else {
                    const int patch = token - 1;
                    const int patch_row = patch / patches_w;
                    const int patch_col = patch % patches_w;
                    value = bias.empty() ? 0.0F : bias[d];

                    int flat_patch = 0;
                    for (int c = 0; c < shape.in_channels; ++c) {
                        for (int i = 0; i < shape.patch_size; ++i) {
                            for (int j = 0; j < shape.patch_size; ++j) {
                                const int row = patch_row * shape.patch_size + i;
                                const int col = patch_col * shape.patch_size + j;
                                const std::size_t pixel_index =
                                    ((static_cast<std::size_t>(b) * shape.in_channels + c) *
                                         shape.height +
                                     row) *
                                        shape.width +
                                    col;
                                value += image[pixel_index] *
                                         weight[static_cast<std::size_t>(flat_patch) *
                                                    shape.d_model +
                                                d];
                                ++flat_patch;
                            }
                        }
                    }
                }

                value += position_embedding[static_cast<std::size_t>(token) *
                                                shape.d_model +
                                            d];
                output[(static_cast<std::size_t>(b) * seq_len + token) *
                           shape.d_model +
                       d] = value;
            }
        }
    }

    return output;
}

void patch_embedding_cuda(
    const float* d_image,
    const float* d_weight,
    const float* d_bias,
    float* d_patch_tokens,
    const PatchEmbeddingShape& shape,
    cudaStream_t stream) {
    shape.validate();
    require_device_pointer(d_image, "d_image");
    require_device_pointer(d_weight, "d_weight");
    require_device_pointer(d_patch_tokens, "d_patch_tokens");

    const int n_patches = shape.n_patches();
    const std::size_t elements =
        static_cast<std::size_t>(n_patches) * shape.d_model;
    patch_embedding_kernel<<<grid_1d(elements, shape.batch), kBlockSize, 0, stream>>>(
        d_image,
        d_weight,
        d_bias,
        d_patch_tokens,
        shape.in_channels,
        shape.height,
        shape.width,
        shape.patch_size,
        shape.patches_w(),
        n_patches,
        shape.d_model);
    check_cuda(cudaGetLastError(), "No se pudo lanzar patch_embedding_kernel");
}

void add_cls_token_cuda(
    const float* d_patch_tokens,
    const float* d_cls_token,
    float* d_tokens_with_cls,
    const PatchEmbeddingShape& shape,
    cudaStream_t stream) {
    shape.validate();
    require_device_pointer(d_patch_tokens, "d_patch_tokens");
    require_device_pointer(d_tokens_with_cls, "d_tokens_with_cls");

    const int n_patches = shape.n_patches();
    const std::size_t elements =
        static_cast<std::size_t>(n_patches + 1) * shape.d_model;
    add_cls_token_kernel<<<grid_1d(elements, shape.batch), kBlockSize, 0, stream>>>(
        d_patch_tokens,
        d_cls_token,
        d_tokens_with_cls,
        n_patches,
        shape.d_model);
    check_cuda(cudaGetLastError(), "No se pudo lanzar add_cls_token_kernel");
}

void add_position_embedding_cuda(
    const float* d_tokens,
    const float* d_position_embedding,
    float* d_output,
    const PatchEmbeddingShape& shape,
    cudaStream_t stream) {
    shape.validate();
    require_device_pointer(d_tokens, "d_tokens");
    require_device_pointer(d_position_embedding, "d_position_embedding");
    require_device_pointer(d_output, "d_output");

    const std::size_t elements =
        static_cast<std::size_t>(shape.seq_len()) * shape.d_model;
    add_position_embedding_kernel<<<grid_1d(elements, shape.batch), kBlockSize, 0, stream>>>(
        d_tokens,
        d_position_embedding,
        d_output,
        shape.seq_len(),
        shape.d_model);
    check_cuda(cudaGetLastError(), "No se pudo lanzar add_position_embedding_kernel");
}

void patch_embedding_pipeline_cuda(
    const float* d_image,
    const float* d_weight,
    const float* d_bias,
    const float* d_cls_token,
    const float* d_position_embedding,
    float* d_output,
    const PatchEmbeddingShape& shape,
    cudaStream_t stream) {
    shape.validate();
    require_device_pointer(d_image, "d_image");
    require_device_pointer(d_weight, "d_weight");
    require_device_pointer(d_position_embedding, "d_position_embedding");
    require_device_pointer(d_output, "d_output");

    const std::size_t elements =
        static_cast<std::size_t>(shape.seq_len()) * shape.d_model;
    patch_embedding_pipeline_kernel<<<grid_1d(elements, shape.batch), kBlockSize, 0, stream>>>(
        d_image,
        d_weight,
        d_bias,
        d_cls_token,
        d_position_embedding,
        d_output,
        shape.in_channels,
        shape.height,
        shape.width,
        shape.patch_size,
        shape.patches_w(),
        shape.n_patches(),
        shape.d_model);
    check_cuda(cudaGetLastError(), "No se pudo lanzar patch_embedding_pipeline_kernel");
}

namespace {

__global__ void patch_embedding_pipeline_backward_kernel(
    const float* d_image,
    const float* d_grad_out,
    float* d_grad_weight,
    float* d_grad_bias,
    float* d_grad_cls_token,
    float* d_grad_pos_embed,
    int batch,
    int channels,
    int height,
    int width,
    int patch_size,
    int patches_w,
    int n_patches,
    int d_model) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int seq_len = n_patches + 1;
    // solo la primera fila de bloques acumula pos/cls/bias: el grid tiene
    // patch_dim filas (blockIdx.y) y sin este guard se acumularia patch_dim veces
    if (blockIdx.y == 0 && d < d_model) {
        for (int t = 0; t < seq_len; ++t) {
            float sum_pos = 0.0f;
            for (int b = 0; b < batch; ++b) {
                sum_pos += d_grad_out[(static_cast<std::size_t>(b) * seq_len + t) * d_model + d];
            }
            if (d_grad_pos_embed != nullptr) {
                atomicAdd(&d_grad_pos_embed[t * d_model + d], sum_pos);
            }
            if (t == 0 && d_grad_cls_token != nullptr) {
                atomicAdd(&d_grad_cls_token[d], sum_pos);
            }
        }

        if (d_grad_bias != nullptr) {
            float sum_b = 0.0f;
            for (int b = 0; b < batch; ++b) {
                for (int p = 0; p < n_patches; ++p) {
                    sum_b += d_grad_out[(static_cast<std::size_t>(b) * seq_len + (p + 1)) * d_model + d];
                }
            }
            atomicAdd(&d_grad_bias[d], sum_b);
        }
    }

    if (d_grad_weight != nullptr) {
        int flat_patch = blockIdx.y;
        int d_idx = d;
        int patch_dim = channels * patch_size * patch_size;
        if (flat_patch < patch_dim && d_idx < d_model) {
            int c = flat_patch / (patch_size * patch_size);
            int rem = flat_patch % (patch_size * patch_size);
            int pi = rem / patch_size;
            int pj = rem % patch_size;

            float acc_w = 0.0f;
            for (int b = 0; b < batch; ++b) {
                for (int p = 0; p < n_patches; ++p) {
                    int patch_row = p / patches_w;
                    int patch_col = p % patches_w;
                    int r = patch_row * patch_size + pi;
                    int col = patch_col * patch_size + pj;
                    float pixel = d_image[((static_cast<std::size_t>(b) * channels + c) * height + r) * width + col];
                    float dy = d_grad_out[(static_cast<std::size_t>(b) * seq_len + (p + 1)) * d_model + d_idx];
                    acc_w += pixel * dy;
                }
            }
            atomicAdd(&d_grad_weight[static_cast<std::size_t>(flat_patch) * d_model + d_idx], acc_w);
        }
    }
}

}  // namespace

void patch_embedding_pipeline_backward_cuda(
    const float* d_image,
    const float* d_grad_out,
    float* d_grad_weight,
    float* d_grad_bias,
    float* d_grad_cls_token,
    float* d_grad_position_embedding,
    const PatchEmbeddingShape& shape,
    cudaStream_t stream) {
    shape.validate();
    require_device_pointer(d_image, "d_image");
    require_device_pointer(d_grad_out, "d_grad_out");

    int patch_dim = shape.patch_dim();
    dim3 grid((shape.d_model + kBlockSize - 1) / kBlockSize, patch_dim);

    patch_embedding_pipeline_backward_kernel<<<grid, kBlockSize, 0, stream>>>(
        d_image,
        d_grad_out,
        d_grad_weight,
        d_grad_bias,
        d_grad_cls_token,
        d_grad_position_embedding,
        shape.batch,
        shape.in_channels,
        shape.height,
        shape.width,
        shape.patch_size,
        shape.patches_w(),
        shape.n_patches(),
        shape.d_model);
    check_cuda(cudaGetLastError(), "No se pudo lanzar patch_embedding_pipeline_backward_kernel");
}

}  // namespace vit
