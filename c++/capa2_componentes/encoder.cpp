#include "encoder.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

using namespace std;

namespace vit {

size_t LayerNormShape::elements() const {
    return (size_t)rows * dim;
}

void LayerNormShape::validate() const {
    if (rows <= 0 || dim <= 0) {
        throw invalid_argument("LayerNormShape: dimensiones deben ser positivas");
    }
}

size_t LinearShape::input_elements() const {
    return (size_t)rows * in_features;
}

size_t LinearShape::output_elements() const {
    return (size_t)rows * out_features;
}

void LinearShape::validate() const {
    if (rows <= 0 || in_features <= 0 || out_features <= 0) {
        throw invalid_argument("LinearShape: dimensiones deben ser positivas");
    }
}


// cada fila es independiente -> paralelizable
void layer_norm_cpu(const float* input, const float* gamma, const float* beta,
                    float* output, const LayerNormShape& shape, float eps) {
    shape.validate();

    for (int r = 0; r < shape.rows; ++r) {
        const float* x = input + (size_t)r * shape.dim;
        float* y = output + (size_t)r * shape.dim;

        float mean = 0.0f;
        for (int i = 0; i < shape.dim; ++i) {
            mean += x[i];
        }
        mean /= (float)shape.dim;

        float var = 0.0f;
        for (int i = 0; i < shape.dim; ++i) {
            float d = x[i] - mean;
            var += d * d;
        }
        var /= (float)shape.dim;

        float inv = 1.0f / sqrt(var + eps);
        for (int i = 0; i < shape.dim; ++i) {
            y[i] = (x[i] - mean) * inv * gamma[i] + beta[i];
        }
    }
}


// cada elemento de salida es independiente -> paralelizable
void linear_cpu(const float* input, const float* weight, const float* bias,
                float* output, const LinearShape& shape) {
    shape.validate();

    for (int r = 0; r < shape.rows; ++r) {
        const float* x = input + (size_t)r * shape.in_features;
        float* y = output + (size_t)r * shape.out_features;

        for (int o = 0; o < shape.out_features; ++o) {
            float acc = bias != nullptr ? bias[o] : 0.0f;
            for (int i = 0; i < shape.in_features; ++i) {
                acc += x[i] * weight[(size_t)i * shape.out_features + o];
            }
            y[o] = acc;
        }
    }
}


void gelu_cpu(const float* input, float* output, size_t n) {
    float k = 0.7978845608028654f;  // sqrt(2/pi)
    float c = 0.044715f;

    for (size_t i = 0; i < n; ++i) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + tanh(k * (x + c * x * x * x)));
    }
}

}  // namespace vit


// ---- LayerNorm ----

LayerNorm::LayerNorm(size_t d_model, float eps)
    : d_model_(d_model),
      eps_(eps),
      gamma_(d_model, 1.0f),
      beta_(d_model, 0.0f) {

    if (d_model_ == 0) {
        throw invalid_argument("LayerNorm: d_model debe ser mayor que cero");
    }
}

Tensor LayerNorm::forward(const Tensor& x) {
    if (x.ndim() != 3 || x.dim(2) != d_model_) {
        throw invalid_argument("LayerNorm espera [batch, tokens, d_model]");
    }

    vit::LayerNormShape shape{(int)(x.dim(0) * x.dim(1)), (int)d_model_};

    Tensor out({x.dim(0), x.dim(1), x.dim(2)}, 0.0f);
    vit::layer_norm_cpu(x.data(), gamma_.data(), beta_.data(), out.data(),
                        shape, eps_);
    return out;
}


// ---- MLP ----

MLP::MLP(size_t d_model, size_t hidden_dim, unsigned seed)
    : d_model_(d_model),
      hidden_dim_(hidden_dim == 0 ? d_model * 4 : hidden_dim) {

    if (d_model_ == 0) {
        throw invalid_argument("MLP: d_model debe ser mayor que cero");
    }

    // xavier uniform, igual que la version python
    mt19937 rng(seed);
    float l1 = sqrt(6.0f / (float)(d_model_ + hidden_dim_));
    float l2 = sqrt(6.0f / (float)(hidden_dim_ + d_model_));
    uniform_real_distribution<float> d1(-l1, l1);
    uniform_real_distribution<float> d2(-l2, l2);

    w1_.resize(d_model_ * hidden_dim_);
    for (float& v : w1_) {
        v = d1(rng);
    }
    b1_.assign(hidden_dim_, 0.0f);

    w2_.resize(hidden_dim_ * d_model_);
    for (float& v : w2_) {
        v = d2(rng);
    }
    b2_.assign(d_model_, 0.0f);
}

size_t MLP::hidden_dim() const {
    return hidden_dim_;
}

Tensor MLP::forward(const Tensor& x) {
    if (x.ndim() != 3 || x.dim(2) != d_model_) {
        throw invalid_argument("MLP espera [batch, tokens, d_model]");
    }

    int rows = (int)(x.dim(0) * x.dim(1));

    vector<float> hidden((size_t)rows * hidden_dim_, 0.0f);
    vit::linear_cpu(x.data(), w1_.data(), b1_.data(), hidden.data(),
                    {rows, (int)d_model_, (int)hidden_dim_});
    vit::gelu_cpu(hidden.data(), hidden.data(), hidden.size());

    Tensor out({x.dim(0), x.dim(1), x.dim(2)}, 0.0f);
    vit::linear_cpu(hidden.data(), w2_.data(), b2_.data(), out.data(),
                    {rows, (int)hidden_dim_, (int)d_model_});
    return out;
}


// ---- TransformerEncoderBlock ----

TransformerEncoderBlock::TransformerEncoderBlock(size_t d_model, size_t n_heads,
                                                 size_t mlp_hidden_dim,
                                                 unsigned seed)
    : norm1_(d_model),
      attention_(d_model, n_heads),
      norm2_(d_model),
      mlp_(d_model, mlp_hidden_dim, seed) {}

Tensor TransformerEncoderBlock::forward(const Tensor& x) {
    Tensor a = x + attention_.forward(norm1_.forward(x));  // residual 1
    return a + mlp_.forward(norm2_.forward(a));            // residual 2
}


// ---- TransformerEncoder ----

TransformerEncoder::TransformerEncoder(size_t d_model, size_t n_heads,
                                       size_t n_layers, size_t mlp_hidden_dim,
                                       unsigned seed) {
    if (n_layers == 0) {
        throw invalid_argument("TransformerEncoder: n_layers debe ser mayor que cero");
    }

    blocks_.reserve(n_layers);
    for (size_t i = 0; i < n_layers; ++i) {
        blocks_.emplace_back(d_model, n_heads, mlp_hidden_dim, seed + (unsigned)i);
    }
}

size_t TransformerEncoder::n_layers() const {
    return blocks_.size();
}

Tensor TransformerEncoder::forward(const Tensor& x) {
    Tensor out = x;

    for (TransformerEncoderBlock& block : blocks_) {
        out = block.forward(out);
    }
    return out;
}
