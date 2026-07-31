
#pragma once

#include "../capa1_nucleo/cuda_utils.cuh"

#include <cstddef>
#include <vector>

namespace vit {

// formas de las ops paralelizables
struct LayerNormShape {
    int rows;  // batch * tokens
    int dim;   // d_model

    std::size_t elements() const;
    void validate() const;
};

struct LinearShape {
    int rows;  // batch * tokens
    int in_features;
    int out_features;

    std::size_t input_elements() const;
    std::size_t output_elements() const;
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
void gelu_cpu(const float* input, float* output, std::size_t n);

// ---- CUDA versions ----

// un bloque por fila; normaliza en paralelo
void layer_norm_cuda(const float* d_input, const float* d_gamma,
                     const float* d_beta, float* d_output,
                     const LayerNormShape& shape, float eps,
                     cudaStream_t stream = nullptr);

// un hilo por elemento de salida
void linear_cuda(const float* d_input, const float* d_weight,
                 const float* d_bias, float* d_output,
                 const LinearShape& shape, cudaStream_t stream = nullptr);

// un hilo por elemento
void gelu_cuda(const float* d_input, float* d_output, std::size_t n,
               cudaStream_t stream = nullptr);

// ---- CUDA backward versions ----

void layer_norm_backward_cuda(const float* d_input, const float* d_gamma,
                              const float* d_grad_out, float* d_grad_in,
                              float* d_grad_gamma, float* d_grad_beta,
                              const LayerNormShape& shape, float eps,
                              cudaStream_t stream = nullptr);

void linear_backward_cuda(const float* d_input, const float* d_weight,
                          const float* d_grad_out, float* d_grad_in,
                          float* d_grad_weight, float* d_grad_bias,
                          const LinearShape& shape,
                          cudaStream_t stream = nullptr);

void gelu_backward_cuda(const float* d_input, const float* d_grad_out,
                        float* d_grad_in, std::size_t n,
                        cudaStream_t stream = nullptr);

}  // namespace vit
