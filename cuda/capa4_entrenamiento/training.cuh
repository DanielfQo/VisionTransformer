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

// entrena el ViT en un dataset MNIST
// retorna el accuracy final en test
float train_vit(const float* h_train_images, const int* h_train_labels, int n_train,
                const float* h_test_images, const int* h_test_labels, int n_test,
                VitWeights& weights, const VitConfig& config,
                int epochs, int batch_size, float learning_rate);

}  // namespace vit
