#pragma once

#include "../capa1_nucleo/module.hpp"
#include "attention.hpp"

#include <cstddef>
#include <vector>

using namespace std;

namespace vit {

// formas de las ops paralelizables
struct LayerNormShape {
    int rows;  // batch * tokens
    int dim;   // d_model

    size_t elements() const;
    void validate() const;
};

struct LinearShape {
    int rows;  // batch * tokens
    int in_features;
    int out_features;

    size_t input_elements() const;
    size_t output_elements() const;
    void validate() const;
};

// primitivas cpu que la version cuda paraleliza

// normaliza cada fila y aplica gamma/beta
void layer_norm_cpu(const float* input, const float* gamma, const float* beta,
                    float* output, const LayerNormShape& shape, float eps);

// output = input * weight + bias ; weight en layout [in, out]
void linear_cpu(const float* input, const float* weight, const float* bias,
                float* output, const LinearShape& shape);

// gelu elemento a elemento
void gelu_cpu(const float* input, float* output, size_t n);

}  // namespace vit


// normalizacion por capa sobre el ultimo eje
class LayerNorm : public Module {
public:
    explicit LayerNorm(size_t d_model, float eps = 1e-5f);

    Tensor forward(const Tensor& x) override;

private:
    size_t d_model_;
    float eps_;
    vector<float> gamma_;
    vector<float> beta_;
};


// feed-forward: linear -> gelu -> linear
class MLP : public Module {
public:
    MLP(size_t d_model, size_t hidden_dim = 0, unsigned seed = 0);

    Tensor forward(const Tensor& x) override;

    size_t hidden_dim() const;

private:
    size_t d_model_;
    size_t hidden_dim_;
    vector<float> w1_, b1_, w2_, b2_;
};


// bloque: x + attn(norm1(x)) ; luego + mlp(norm2(x))
class TransformerEncoderBlock : public Module {
public:
    TransformerEncoderBlock(size_t d_model, size_t n_heads,
                            size_t mlp_hidden_dim = 0, unsigned seed = 0);

    Tensor forward(const Tensor& x) override;

private:
    LayerNorm norm1_;
    MultiHeadSelfAttention attention_;
    LayerNorm norm2_;
    MLP mlp_;
};


// apila n bloques
class TransformerEncoder : public Module {
public:
    TransformerEncoder(size_t d_model, size_t n_heads, size_t n_layers,
                       size_t mlp_hidden_dim = 0, unsigned seed = 0);

    Tensor forward(const Tensor& x) override;

    size_t n_layers() const;

private:
    vector<TransformerEncoderBlock> blocks_;
};
