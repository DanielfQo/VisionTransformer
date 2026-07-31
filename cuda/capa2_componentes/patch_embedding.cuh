#pragma once

#include "../capa1_nucleo/cuda_utils.cuh"

#include <cstddef>
#include <vector>

namespace vit {

struct PatchEmbeddingShape {
    int batch;
    int in_channels;
    int height;
    int width;
    int patch_size;
    int d_model;

    int patches_h() const;
    int patches_w() const;
    int n_patches() const;
    int patch_dim() const;
    int seq_len() const;
    std::size_t image_elements() const;
    std::size_t patch_token_elements() const;
    std::size_t output_elements() const;
    std::size_t weight_elements() const;
    std::size_t position_elements() const;
    void validate() const;
};

std::vector<float> patch_embedding_pipeline_cpu(
    const std::vector<float>& image,
    const std::vector<float>& weight,
    const std::vector<float>& bias,
    const std::vector<float>& cls_token,
    const std::vector<float>& position_embedding,
    const PatchEmbeddingShape& shape);

void patch_embedding_cuda(
    const float* d_image,
    const float* d_weight,
    const float* d_bias,
    float* d_patch_tokens,
    const PatchEmbeddingShape& shape,
    cudaStream_t stream = nullptr);

void add_cls_token_cuda(
    const float* d_patch_tokens,
    const float* d_cls_token,
    float* d_tokens_with_cls,
    const PatchEmbeddingShape& shape,
    cudaStream_t stream = nullptr);

void add_position_embedding_cuda(
    const float* d_tokens,
    const float* d_position_embedding,
    float* d_output,
    const PatchEmbeddingShape& shape,
    cudaStream_t stream = nullptr);

void patch_embedding_pipeline_cuda(
    const float* d_image,
    const float* d_weight,
    const float* d_bias,
    const float* d_cls_token,
    const float* d_position_embedding,
    float* d_output,
    const PatchEmbeddingShape& shape,
    cudaStream_t stream = nullptr);

}  // namespace vit
