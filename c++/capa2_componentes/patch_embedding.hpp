#ifndef CAPA2_COMPONENTES_PATCH_EMBEDDING_HPP
#define CAPA2_COMPONENTES_PATCH_EMBEDDING_HPP

#include "../capa1_nucleo/module.hpp"

#include <cstddef>
#include <vector>

class PatchEmbedding : public Module {
public:
    PatchEmbedding(std::size_t in_channels,
                   std::size_t patch_size,
                   std::size_t d_model,
                   unsigned int seed = 42);

    Tensor forward(const Tensor& x) override;

    std::size_t in_channels() const;
    std::size_t patch_size() const;
    std::size_t d_model() const;
    std::size_t patch_dim() const;

private:
    std::size_t in_channels_;
    std::size_t patch_size_;
    std::size_t d_model_;
    std::size_t patch_dim_;
    std::vector<float> weight_;
    std::vector<float> bias_;

    float project_value(const Tensor& x,
                        std::size_t batch,
                        std::size_t patch_row,
                        std::size_t patch_col,
                        std::size_t out_dim) const;
};

class CLSToken : public Module {
public:
    explicit CLSToken(std::size_t d_model);

    Tensor forward(const Tensor& x) override;

private:
    std::size_t d_model_;
    std::vector<float> cls_token_;
};

class PositionEmbedding : public Module {
public:
    PositionEmbedding(std::size_t seq_len,
                      std::size_t d_model,
                      unsigned int seed = 42);

    Tensor forward(const Tensor& x) override;

private:
    std::size_t seq_len_;
    std::size_t d_model_;
    std::vector<float> pos_embedding_;
};

class PatchEmbeddingPipeline : public Module {
public:
    PatchEmbeddingPipeline(std::size_t in_channels,
                           std::size_t image_size,
                           std::size_t patch_size,
                           std::size_t d_model,
                           unsigned int seed = 42);

    Tensor forward(const Tensor& x) override;

    std::size_t n_patches() const;
    std::size_t seq_len() const;

private:
    std::size_t image_size_;
    std::size_t patch_size_;
    std::size_t n_patches_;
    std::size_t seq_len_;

    PatchEmbedding patch_embed_;
    CLSToken cls_token_;
    PositionEmbedding pos_embed_;
};

#endif
