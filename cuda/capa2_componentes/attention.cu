#include "attention.cuh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace vit {

// ---- AttentionShape methods ----

int AttentionShape::head_dim() const {
    return d_model / heads;
}

std::size_t AttentionShape::token_elements() const {
    return static_cast<std::size_t>(batch) * tokens * d_model;
}

std::size_t AttentionShape::weight_elements() const {
    return static_cast<std::size_t>(batch) * heads * tokens * tokens;
}

void AttentionShape::validate() const {
    if (batch <= 0 || tokens <= 0 || d_model <= 0 || heads <= 0) {
        throw std::invalid_argument("Las dimensiones de attention deben ser positivas");
    }
    if (d_model % heads != 0) {
        throw std::invalid_argument("d_model debe ser divisible entre heads");
    }
}

// ---- CPU reference ----

std::vector<float> multi_head_self_attention_cpu(
    const std::vector<float>& input,
    const AttentionShape& shape,
    const std::vector<std::uint8_t>* mask,
    std::vector<float>* attention_weights) {
    shape.validate();
    if (input.size() != shape.token_elements()) {
        throw std::invalid_argument("El tamano de input no coincide con AttentionShape");
    }
    if (mask != nullptr && mask->size() != shape.weight_elements()) {
        throw std::invalid_argument("La mascara debe tener forma [batch, heads, tokens, tokens]");
    }

    std::vector<float> output(shape.token_elements(), 0.0F);
    if (attention_weights != nullptr) {
        attention_weights->assign(shape.weight_elements(), 0.0F);
    }

    const int dim = shape.head_dim();
    const float scale = 1.0F / std::sqrt(static_cast<float>(dim));
    std::vector<float> scores(shape.tokens);

    for (int b = 0; b < shape.batch; ++b) {
        for (int h = 0; h < shape.heads; ++h) {
            const int feature = h * dim;
            for (int q = 0; q < shape.tokens; ++q) {
                float maximum = -std::numeric_limits<float>::infinity();
                bool any_allowed = false;

                for (int k = 0; k < shape.tokens; ++k) {
                    const std::size_t weight_index =
                        ((static_cast<std::size_t>(b) * shape.heads + h) *
                             shape.tokens +
                         q) *
                            shape.tokens +
                        k;
                    const bool allowed = mask == nullptr || (*mask)[weight_index] != 0;
                    any_allowed = any_allowed || allowed;

                    float dot = 0.0F;
                    for (int d = 0; d < dim; ++d) {
                        const std::size_t q_index =
                            (static_cast<std::size_t>(b) * shape.tokens + q) *
                                shape.d_model +
                            feature + d;
                        const std::size_t k_index =
                            (static_cast<std::size_t>(b) * shape.tokens + k) *
                                shape.d_model +
                            feature + d;
                        dot += input[q_index] * input[k_index];
                    }
                    scores[k] = allowed ? dot * scale
                                        : -std::numeric_limits<float>::infinity();
                    maximum = std::max(maximum, scores[k]);
                }

                if (!any_allowed) {
                    continue;
                }

                float denominator = 0.0F;
                for (int k = 0; k < shape.tokens; ++k) {
                    scores[k] = std::exp(scores[k] - maximum);
                    denominator += scores[k];
                }

                for (int k = 0; k < shape.tokens; ++k) {
                    const float w = scores[k] / denominator;
                    if (attention_weights != nullptr) {
                        const std::size_t weight_index =
                            ((static_cast<std::size_t>(b) * shape.heads + h) *
                                 shape.tokens +
                             q) *
                                shape.tokens +
                            k;
                        (*attention_weights)[weight_index] = w;
                    }
                    for (int d = 0; d < dim; ++d) {
                        const std::size_t output_index =
                            (static_cast<std::size_t>(b) * shape.tokens + q) *
                                shape.d_model +
                            feature + d;
                        const std::size_t value_index =
                            (static_cast<std::size_t>(b) * shape.tokens + k) *
                                shape.d_model +
                            feature + d;
                        output[output_index] += w * input[value_index];
                    }
                }
            }
        }
    }
    return output;
}

namespace {

__global__ void attention_kernel(
    const float* input,
    float* output,
    const std::uint8_t* mask,
    float* weights,
    int tokens,
    int d_model,
    int heads) {
    const int q = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    const int dim = d_model / heads;
    const int feature = h * dim;
    const std::size_t row =
        (static_cast<std::size_t>(b) * heads + h) * tokens + q;

    // memoria compartida dinamica para cachear los scores de atencion (Q.K)
    extern __shared__ float s_scores[];

    // Paso 1: calcular Q.K solo UNA vez y guardar en s_scores[k]
    float local_max = -CUDART_INF_F;
    for (int k = threadIdx.x; k < tokens; k += blockDim.x) {
        const std::size_t weight_index = row * tokens + k;
        if (mask != nullptr && mask[weight_index] == 0) {
            s_scores[k] = -CUDART_INF_F;
            continue;
        }

        float dot = 0.0F;
        for (int d = 0; d < dim; ++d) {
            const std::size_t q_index =
                (static_cast<std::size_t>(b) * tokens + q) * d_model + feature + d;
            const std::size_t k_index =
                (static_cast<std::size_t>(b) * tokens + k) * d_model + feature + d;
            dot += input[q_index] * input[k_index];
        }
        float score = dot * rsqrtf(static_cast<float>(dim));
        s_scores[k] = score;
        local_max = fmaxf(local_max, score);
    }
    __syncthreads();

    float max_val = block_reduce_max(local_max);
    __shared__ float s_max;
    if (threadIdx.x == 0) {
        s_max = max_val;
    }
    __syncthreads();
    max_val = s_max;

    // Paso 2: denominador exp utilizando s_scores[k] de la memoria compartida
    float local_sum = 0.0F;
    if (max_val != -CUDART_INF_F) {
        for (int k = threadIdx.x; k < tokens; k += blockDim.x) {
            float score = s_scores[k];
            if (score != -CUDART_INF_F) {
                local_sum += expf(score - max_val);
            }
        }
    }

    float denom = block_reduce_sum(local_sum);
    __shared__ float s_denom;
    if (threadIdx.x == 0) {
        s_denom = denom;
    }
    __syncthreads();
    denom = s_denom;

    // Paso 3: acumulacion O = P * V reutilizando s_scores[k] de memoria compartida
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float acc = 0.0F;
        if (denom > 0.0F && max_val != -CUDART_INF_F) {
            for (int k = 0; k < tokens; ++k) {
                float score = s_scores[k];
                if (score == -CUDART_INF_F) {
                    continue;
                }
                float w = expf(score - max_val) / denom;

                if (weights != nullptr && d == 0) {
                    weights[row * tokens + k] = w;
                }

                const std::size_t v_index =
                    (static_cast<std::size_t>(b) * tokens + k) * d_model + feature + d;
                acc += w * input[v_index];
            }
        }

        const std::size_t output_index =
            (static_cast<std::size_t>(b) * tokens + q) * d_model + feature + d;
        output[output_index] = acc;
    }
}

// backward exacto de softmax(Q.K^T/sqrt(dh)).V con Q=K=V=Y.
// un bloque por (head, batch); A y dA caben en memoria compartida (tokens <= 64).
__global__ void attention_backward_kernel(
    const float* input,      // Y [B, S, D]
    const float* probs,      // A [B, H, S, S]
    const float* grad_out,   // dO [B, S, D]
    float* grad_in,          // dY [B, S, D]
    int tokens,
    int d_model,
    int heads) {
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int dim = d_model / heads;
    const int feature = h * dim;
    const float scale = rsqrtf(static_cast<float>(dim));
    const int n2 = tokens * tokens;

    extern __shared__ float smem[];
    float* s_a    = smem;            // A            [tokens, tokens]
    float* s_da   = smem + n2;       // dA, luego dS [tokens, tokens]
    float* s_row  = s_da + n2;       // sum_j A*dA por fila [tokens]

    const std::size_t probs_base =
        (static_cast<std::size_t>(b) * heads + h) * n2;

    // indice del elemento (t, d) de este head en un tensor [B, S, D]
    auto elem = [&](int t, int d) {
        return (static_cast<std::size_t>(b) * tokens + t) * d_model + feature + d;
    };

    // fase 0: cargar A en memoria compartida
    for (int t = threadIdx.x; t < n2; t += blockDim.x) {
        s_a[t] = probs[probs_base + t];
    }
    __syncthreads();

    // fase 1: dV[j,d] = sum_i A[i,j] * dO[i,d]  (primera contribucion a dY)
    for (int t = threadIdx.x; t < tokens * dim; t += blockDim.x) {
        const int j = t / dim;
        const int d = t % dim;
        float acc = 0.0f;
        for (int i = 0; i < tokens; ++i) {
            acc += s_a[i * tokens + j] * grad_out[elem(i, d)];
        }
        grad_in[elem(j, d)] = acc;
    }

    // fase 2: dA[i,j] = sum_d dO[i,d] * V[j,d]   (V = input)
    for (int t = threadIdx.x; t < n2; t += blockDim.x) {
        const int i = t / tokens;
        const int j = t % tokens;
        float acc = 0.0f;
        for (int d = 0; d < dim; ++d) {
            acc += grad_out[elem(i, d)] * input[elem(j, d)];
        }
        s_da[t] = acc;
    }
    __syncthreads();

    // fase 3: s_row[i] = sum_j A[i,j] * dA[i,j]
    for (int i = threadIdx.x; i < tokens; i += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < tokens; ++j) {
            acc += s_a[i * tokens + j] * s_da[i * tokens + j];
        }
        s_row[i] = acc;
    }
    __syncthreads();

    // fase 4: softmax backward + escala: dS[i,j] = A[i,j]*(dA[i,j] - s_row[i])/sqrt(dh)
    for (int t = threadIdx.x; t < n2; t += blockDim.x) {
        const int i = t / tokens;
        s_da[t] = s_a[t] * (s_da[t] - s_row[i]) * scale;
    }
    __syncthreads();

    // fase 5: dY[t,d] += dQ[t,d] + dK[t,d]
    //   dQ[t,d] = sum_j dS[t,j] * K[j,d]   (K = input)
    //   dK[t,d] = sum_j dS[j,t] * Q[j,d]   (Q = input)
    for (int t = threadIdx.x; t < tokens * dim; t += blockDim.x) {
        const int i = t / dim;
        const int d = t % dim;
        float acc = 0.0f;
        for (int j = 0; j < tokens; ++j) {
            const float y = input[elem(j, d)];
            acc += s_da[i * tokens + j] * y;   // dQ
            acc += s_da[j * tokens + i] * y;   // dK
        }
        grad_in[elem(i, d)] += acc;
    }
}

}  // namespace

void multi_head_self_attention_cuda(
    const float* d_input,
    float* d_output,
    const AttentionShape& shape,
    const std::uint8_t* d_mask,
    float* d_attention_weights,
    cudaStream_t stream) {
    shape.validate();
    if (d_input == nullptr || d_output == nullptr) {
        throw std::invalid_argument("d_input y d_output no pueden ser nullptr");
    }

    const dim3 grid(shape.tokens, shape.heads, shape.batch);
    std::size_t smem_bytes = static_cast<std::size_t>(shape.tokens) * sizeof(float);
    attention_kernel<<<grid, kBlockSize, smem_bytes, stream>>>(
        d_input,
        d_output,
        d_mask,
        d_attention_weights,
        shape.tokens,
        shape.d_model,
        shape.heads);
    check_cuda(cudaGetLastError(), "No se pudo lanzar attention_kernel");
}

void multi_head_self_attention_backward_cuda(
    const float* d_input,
    const float* d_probs,
    const float* d_grad_out,
    float* d_grad_in,
    const AttentionShape& shape,
    cudaStream_t stream) {
    shape.validate();
    if (d_input == nullptr || d_probs == nullptr ||
        d_grad_out == nullptr || d_grad_in == nullptr) {
        throw std::invalid_argument(
            "multi_head_self_attention_backward_cuda: punteros nulos");
    }
    if (shape.tokens > 64) {
        throw std::invalid_argument(
            "attention backward: tokens > 64 excede la memoria compartida");
    }

    const int n2 = shape.tokens * shape.tokens;
    std::size_t smem_bytes =
        static_cast<std::size_t>(2 * n2 + shape.tokens) * sizeof(float);
    const dim3 grid(shape.heads, shape.batch);
    attention_backward_kernel<<<grid, kBlockSize, smem_bytes, stream>>>(
        d_input,
        d_probs,
        d_grad_out,
        d_grad_in,
        shape.tokens,
        shape.d_model,
        shape.heads);
    check_cuda(cudaGetLastError(), "No se pudo lanzar attention_backward_kernel");
}

}  // namespace vit
