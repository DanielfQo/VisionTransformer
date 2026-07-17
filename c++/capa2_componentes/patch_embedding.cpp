#include "patch_embedding.hpp"

#include <cmath>
#include <random>
#include <stdexcept>
#include <string>

namespace {
void require_rank(const Tensor& x, std::size_t expected, const char* name) {
    if (x.ndim() != expected) {
        throw std::invalid_argument(std::string(name) + " expects rank " +
                                    std::to_string(expected));
    }
}

std::size_t flat3(std::size_t b,
                  std::size_t n,
                  std::size_t d,
                  std::size_t seq_len,
                  std::size_t d_model) {
    return (b * seq_len + n) * d_model + d;
}
}  // namespace

PatchEmbedding::PatchEmbedding(std::size_t in_channels,
                               std::size_t patch_size,
                               std::size_t d_model,
                               unsigned int seed)
    : in_channels_(in_channels),
      patch_size_(patch_size),
      d_model_(d_model),
      patch_dim_(in_channels * patch_size * patch_size),
      weight_(patch_dim_ * d_model),
      bias_(d_model, 0.0f) {
    if (in_channels_ == 0 || patch_size_ == 0 || d_model_ == 0) {
        throw std::invalid_argument("PatchEmbedding dimensions must be greater than zero");
    }

    const float limit = std::sqrt(6.0f / static_cast<float>(patch_dim_ + d_model_));
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-limit, limit);

    for (float& value : weight_) {
        value = dist(rng);
    }
}

Tensor PatchEmbedding::forward(const Tensor& x) {
    require_rank(x, 4, "PatchEmbedding");

    const std::size_t batch_size = x.dim(0);
    const std::size_t channels = x.dim(1);
    const std::size_t height = x.dim(2);
    const std::size_t width = x.dim(3);

    if (channels != in_channels_) {
        throw std::invalid_argument("PatchEmbedding received an unexpected channel count");
    }
    if (height % patch_size_ != 0 || width % patch_size_ != 0) {
        throw std::invalid_argument("Image height and width must be divisible by patch_size");
    }

    const std::size_t patches_h = height / patch_size_;
    const std::size_t patches_w = width / patch_size_;
    const std::size_t n_patches = patches_h * patches_w;

    Tensor out({batch_size, n_patches, d_model_}, 0.0f);

    for (std::size_t b = 0; b < batch_size; ++b) {
        for (std::size_t pr = 0; pr < patches_h; ++pr) {
            for (std::size_t pc = 0; pc < patches_w; ++pc) {
                const std::size_t patch_index = pr * patches_w + pc;
                for (std::size_t d = 0; d < d_model_; ++d) {
                    out.data()[flat3(b, patch_index, d, n_patches, d_model_)] =
                        project_value(x, b, pr, pc, d);
                }
            }
        }
    }

    return out;
}

std::size_t PatchEmbedding::in_channels() const {
    return in_channels_;
}

std::size_t PatchEmbedding::patch_size() const {
    return patch_size_;
}

std::size_t PatchEmbedding::d_model() const {
    return d_model_;
}

std::size_t PatchEmbedding::patch_dim() const {
    return patch_dim_;
}

float PatchEmbedding::project_value(const Tensor& x,
                                    std::size_t batch,
                                    std::size_t patch_row,
                                    std::size_t patch_col,
                                    std::size_t out_dim) const {
    float sum = bias_[out_dim];
    std::size_t flat_patch_index = 0;

    for (std::size_t c = 0; c < in_channels_; ++c) {
        for (std::size_t i = 0; i < patch_size_; ++i) {
            for (std::size_t j = 0; j < patch_size_; ++j) {
                const std::size_t image_row = patch_row * patch_size_ + i;
                const std::size_t image_col = patch_col * patch_size_ + j;
                const float pixel = x.at({batch, c, image_row, image_col});
                sum += pixel * weight_[flat_patch_index * d_model_ + out_dim];
                ++flat_patch_index;
            }
        }
    }

    return sum;
}

CLSToken::CLSToken(std::size_t d_model)
    : d_model_(d_model), cls_token_(d_model, 0.0f) {
    if (d_model_ == 0) {
        throw std::invalid_argument("CLSToken d_model must be greater than zero");
    }
}

Tensor CLSToken::forward(const Tensor& x) {
    require_rank(x, 3, "CLSToken");

    const std::size_t batch_size = x.dim(0);
    const std::size_t n_tokens = x.dim(1);
    const std::size_t d_model = x.dim(2);

    if (d_model != d_model_) {
        throw std::invalid_argument("CLSToken received an unexpected d_model");
    }

    Tensor out({batch_size, n_tokens + 1, d_model_}, 0.0f);

    for (std::size_t b = 0; b < batch_size; ++b) {
        for (std::size_t d = 0; d < d_model_; ++d) {
            out.data()[flat3(b, 0, d, n_tokens + 1, d_model_)] = cls_token_[d];
        }

        for (std::size_t n = 0; n < n_tokens; ++n) {
            for (std::size_t d = 0; d < d_model_; ++d) {
                out.data()[flat3(b, n + 1, d, n_tokens + 1, d_model_)] =
                    x.at({b, n, d});
            }
        }
    }

    return out;
}

PositionEmbedding::PositionEmbedding(std::size_t seq_len,
                                     std::size_t d_model,
                                     unsigned int seed)
    : seq_len_(seq_len),
      d_model_(d_model),
      pos_embedding_(seq_len * d_model) {
    if (seq_len_ == 0 || d_model_ == 0) {
        throw std::invalid_argument("PositionEmbedding dimensions must be greater than zero");
    }

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.02f);
    for (float& value : pos_embedding_) {
        value = dist(rng);
    }
}

Tensor PositionEmbedding::forward(const Tensor& x) {
    require_rank(x, 3, "PositionEmbedding");

    const std::size_t batch_size = x.dim(0);
    const std::size_t seq_len = x.dim(1);
    const std::size_t d_model = x.dim(2);

    if (seq_len != seq_len_ || d_model != d_model_) {
        throw std::invalid_argument("PositionEmbedding received an unexpected shape");
    }

    Tensor out({batch_size, seq_len_, d_model_}, 0.0f);

    for (std::size_t b = 0; b < batch_size; ++b) {
        for (std::size_t n = 0; n < seq_len_; ++n) {
            for (std::size_t d = 0; d < d_model_; ++d) {
                out.data()[flat3(b, n, d, seq_len_, d_model_)] =
                    x.at({b, n, d}) + pos_embedding_[n * d_model_ + d];
            }
        }
    }

    return out;
}

PatchEmbeddingPipeline::PatchEmbeddingPipeline(std::size_t in_channels,
                                               std::size_t image_size,
                                               std::size_t patch_size,
                                               std::size_t d_model,
                                               unsigned int seed)
    : image_size_(image_size),
      patch_size_(patch_size),
      n_patches_(0),
      seq_len_(0),
      patch_embed_(in_channels, patch_size, d_model, seed),
      cls_token_(d_model),
      pos_embed_(1, d_model, seed + 1) {
    if (image_size_ == 0 || patch_size_ == 0) {
        throw std::invalid_argument("image_size and patch_size must be greater than zero");
    }
    if (image_size_ % patch_size_ != 0) {
        throw std::invalid_argument("image_size must be divisible by patch_size");
    }

    const std::size_t patches_per_side = image_size_ / patch_size_;
    n_patches_ = patches_per_side * patches_per_side;
    seq_len_ = n_patches_ + 1;
    pos_embed_ = PositionEmbedding(seq_len_, d_model, seed + 1);
}

Tensor PatchEmbeddingPipeline::forward(const Tensor& x) {
    require_rank(x, 4, "PatchEmbeddingPipeline");

    if (x.dim(2) != image_size_ || x.dim(3) != image_size_) {
        throw std::invalid_argument("PatchEmbeddingPipeline received an unexpected image size");
    }

    Tensor tokens = patch_embed_.forward(x);
    tokens = cls_token_.forward(tokens);
    tokens = pos_embed_.forward(tokens);
    return tokens;
}

std::size_t PatchEmbeddingPipeline::n_patches() const {
    return n_patches_;
}

std::size_t PatchEmbeddingPipeline::seq_len() const {
    return seq_len_;
}
