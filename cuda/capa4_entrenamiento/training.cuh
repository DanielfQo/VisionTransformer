#pragma once

#include "../capa1_nucleo/cuda_utils.cuh"
#include "../capa3_modelo/vit.cuh"

#include <cstddef>

namespace vit {

// calcula cross-entropy loss promedio sobre un batch
// d_logits:  [batch, n_classes]  (logits crudos, sin softmax)
// d_labels:  [batch]             (etiquetas enteras 0..n_classes-1)
// devuelve el loss promedio en CPU
float cross_entropy_loss(const float* d_logits, const int* d_labels,
                         int batch, int n_classes,
                         float* d_loss_workspace,
                         cudaStream_t stream = nullptr);

// d_grad_logits[b, c] = (softmax(logits)[b, c] - (c == labels[b])) / batch
void cross_entropy_backward_cuda(const float* d_logits, const int* d_labels,
                                 float* d_grad_logits, int batch, int n_classes,
                                 cudaStream_t stream = nullptr);

// SGD update in-place: param -= lr * grad
void sgd_update_cuda(float* d_param, const float* d_grad,
                     float lr, std::size_t n,
                     cudaStream_t stream = nullptr);

// ---- optimizador Adam / AdamW ----

struct AdamConfig {
    float beta1 = 0.9f;          // decaimiento del primer momento
    float beta2 = 0.999f;        // decaimiento del segundo momento
    float eps   = 1e-8f;         // estabilidad numerica en el denominador
    float weight_decay = 0.0f;   // AdamW (decoupled); 0 = Adam clasico
};

// estado del optimizador: los momentos m y v tienen la misma forma que los pesos,
// por eso reutilizan la estructura VitWeights
struct AdamState {
    VitWeights m{};
    VitWeights v{};
    long long step = 0;   // numero de actualizaciones aplicadas (para bias correction)
};

void adam_init(AdamState& state, const VitConfig& config);
void adam_free(AdamState& state, const VitConfig& config);

// actualiza un tensor:
//   m = b1*m + (1-b1)*g ; v = b2*v + (1-b2)*g^2
//   param -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * param)
// bias_correction1 = 1 - b1^t, bias_correction2 = 1 - b2^t
void adam_update_cuda(float* d_param, const float* d_grad,
                      float* d_m, float* d_v,
                      float lr, const AdamConfig& cfg,
                      float bias_correction1, float bias_correction2,
                      std::size_t n, cudaStream_t stream = nullptr);

// aplica Adam a todos los parametros del modelo e incrementa state.step
void vit_update_weights_adam(VitWeights& weights, const VitGradients& grads,
                             AdamState& state, const VitConfig& config,
                             float lr, const AdamConfig& cfg,
                             cudaStream_t stream = nullptr);

// ---- scheduler de learning rate ----

// warmup lineal durante warmup_steps y despues decaimiento coseno hasta min_lr.
// el warmup evita que los primeros pasos (con momentos aun sin estimar) desestabilicen
// el modelo; el coseno reduce el paso al final para que converja en vez de oscilar.
float lr_schedule(long long step, long long warmup_steps, long long total_steps,
                  float base_lr, float min_lr);

// ---- configuracion de entrenamiento ----

struct TrainConfig {
    int epochs       = 30;
    int batch_size   = 128;
    float lr         = 1e-3f;    // learning rate pico (al terminar el warmup)
    float min_lr     = 1e-5f;    // learning rate final del coseno
    float warmup_ratio = 0.05f;  // fraccion de los pasos totales dedicada al warmup
    bool use_adam    = true;     // false = SGD plano, sin momentos ni scheduler
    AdamConfig adam{};
};

// entrena el ViT en un dataset MNIST con set de validacion
// retorna el accuracy final en test
float train_vit(const float* h_train_images, const int* h_train_labels, int n_train,
                const float* h_val_images, const int* h_val_labels, int n_val,
                const float* h_test_images, const int* h_test_labels, int n_test,
                VitWeights& weights, const VitConfig& config,
                const TrainConfig& train_cfg);

}  // namespace vit
