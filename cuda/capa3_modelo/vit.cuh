#pragma once

#include "../capa1_nucleo/cuda_utils.cuh"
#include "../capa2_componentes/patch_embedding.cuh"
#include "../capa2_componentes/attention.cuh"
#include "../capa2_componentes/encoder.cuh"
#include "../capa2_componentes/softmax.cuh"

#include <cstddef>

namespace vit {

// ---- configuracion del modelo ----

struct VitConfig {
    int image_size   = 28;
    int patch_size   = 7;
    int in_channels  = 1;
    int d_model      = 64;
    int heads        = 4;
    int n_layers     = 2;
    int mlp_hidden   = 256;
    int n_classes    = 10;
    float eps        = 1e-5f;  // epsilon de LayerNorm

    int n_patches() const { return (image_size / patch_size) * (image_size / patch_size); }
    int seq_len()   const { return n_patches() + 1; }
    int patch_dim() const { return in_channels * patch_size * patch_size; }
};

// ---- pesos del modelo en GPU ----

struct VitWeights {
    // patch embedding
    float* pe_weight;            // [patch_dim, d_model]
    float* pe_bias;              // [d_model]
    float* cls_token;            // [d_model]
    float* position_embedding;   // [seq_len, d_model]

    // encoder layers (arrays de punteros, uno por capa)
    // norm1: gamma, beta
    float** ln1_gamma;           // [n_layers][d_model]
    float** ln1_beta;            // [n_layers][d_model]
    // norm2: gamma, beta
    float** ln2_gamma;           // [n_layers][d_model]
    float** ln2_beta;            // [n_layers][d_model]
    // mlp: w1, b1, w2, b2
    float** mlp_w1;              // [n_layers][d_model * mlp_hidden]
    float** mlp_b1;              // [n_layers][mlp_hidden]
    float** mlp_w2;              // [n_layers][mlp_hidden * d_model]
    float** mlp_b2;              // [n_layers][d_model]

    // final layer norm
    float* final_ln_gamma;       // [d_model]
    float* final_ln_beta;        // [d_model]

    // classification head
    float* head_weight;          // [d_model, n_classes]
    float* head_bias;            // [n_classes]
};

// inicializa pesos con Xavier uniform en CPU, copia a GPU
void vit_init_weights(VitWeights& weights, const VitConfig& config, unsigned seed = 42);

// libera toda la memoria GPU
void vit_free_weights(VitWeights& weights, const VitConfig& config);

// forward pass completo: imagen -> logits (batch=1 por ahora expandible)
// d_image: [batch, channels, height, width] en GPU
// d_logits: [batch, n_classes] en GPU (salida)
// d_workspace debe tener al menos vit_workspace_bytes(config, batch) bytes
void vit_forward(const float* d_image, float* d_logits,
                 float* d_workspace,
                 const VitWeights& weights,
                 const VitConfig& config,
                 int batch,
                 cudaStream_t stream = nullptr);

// calcula los bytes necesarios para el workspace temporal
std::size_t vit_workspace_bytes(const VitConfig& config, int batch);

// ---- gradientes y backward pass ----

using VitGradients = VitWeights;

void vit_init_gradients(VitGradients& grads, const VitConfig& config);
void vit_zero_gradients(VitGradients& grads, const VitConfig& config, cudaStream_t stream = nullptr);
void vit_free_gradients(VitGradients& grads, const VitConfig& config);

void vit_backward(const float* d_image,
                  const float* d_grad_logits,
                  float* d_workspace,
                  const VitWeights& weights,
                  VitGradients& grads,
                  const VitConfig& config,
                  int batch,
                  cudaStream_t stream = nullptr);

void vit_update_weights_sgd(VitWeights& weights,
                            const VitGradients& grads,
                            const VitConfig& config,
                            float learning_rate,
                            cudaStream_t stream = nullptr);

}  // namespace vit
