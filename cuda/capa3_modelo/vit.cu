#include "vit.cuh"
#include "../capa4_entrenamiento/training.cuh"

#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

namespace vit {

// ---- helpers ----

namespace {

// alloca en GPU y copia datos desde host
float* alloc_and_copy(const float* host_data, std::size_t n) {
    float* d_ptr = nullptr;
    std::size_t bytes = n * sizeof(float);
    check_cuda(cudaMalloc(&d_ptr, bytes), "cudaMalloc");
    check_cuda(cudaMemcpy(d_ptr, host_data, bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy H->D");
    return d_ptr;
}

float* alloc_gpu(std::size_t n) {
    float* d_ptr = nullptr;
    check_cuda(cudaMalloc(&d_ptr, n * sizeof(float)), "cudaMalloc");
    check_cuda(cudaMemset(d_ptr, 0, n * sizeof(float)), "cudaMemset");
    return d_ptr;
}

void xavier_uniform(float* data, std::size_t fan_in, std::size_t fan_out,
                     std::size_t n, std::mt19937& rng) {
    float limit = std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));
    std::uniform_real_distribution<float> dist(-limit, limit);
    for (std::size_t i = 0; i < n; ++i) {
        data[i] = dist(rng);
    }
}

// kernel para residual: output = a + b
__global__ void add_kernel(const float* a, const float* b, float* output,
                           std::size_t n) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = a[i] + b[i];
    }
}

void add_cuda(const float* a, const float* b, float* output,
              std::size_t n, cudaStream_t stream) {
    std::size_t blocks = (n + kBlockSize - 1) / kBlockSize;
    add_kernel<<<blocks, kBlockSize, 0, stream>>>(a, b, output, n);
    check_cuda(cudaGetLastError(), "add_kernel");
}

}  // namespace

// ---- init weights ----

void vit_init_weights(VitWeights& weights, const VitConfig& config, unsigned seed) {
    std::mt19937 rng(seed);

    const int d = config.d_model;
    const int pd = config.patch_dim();
    const int sl = config.seq_len();
    const int mh = config.mlp_hidden;
    const int nc = config.n_classes;
    const int nl = config.n_layers;

    // patch embedding
    {
        std::vector<float> w(static_cast<std::size_t>(pd) * d);
        xavier_uniform(w.data(), pd, d, w.size(), rng);
        weights.pe_weight = alloc_and_copy(w.data(), w.size());

        std::vector<float> b(d, 0.0f);
        weights.pe_bias = alloc_and_copy(b.data(), b.size());

        std::vector<float> cls(d, 0.0f);
        xavier_uniform(cls.data(), 1, d, cls.size(), rng);
        weights.cls_token = alloc_and_copy(cls.data(), cls.size());

        std::vector<float> pe(static_cast<std::size_t>(sl) * d);
        xavier_uniform(pe.data(), sl, d, pe.size(), rng);
        weights.position_embedding = alloc_and_copy(pe.data(), pe.size());
    }

    // encoder layers
    weights.ln1_gamma = new float*[nl];
    weights.ln1_beta  = new float*[nl];
    weights.ln2_gamma = new float*[nl];
    weights.ln2_beta  = new float*[nl];
    weights.mlp_w1    = new float*[nl];
    weights.mlp_b1    = new float*[nl];
    weights.mlp_w2    = new float*[nl];
    weights.mlp_b2    = new float*[nl];

    for (int l = 0; l < nl; ++l) {
        // layer norm 1
        std::vector<float> gamma(d, 1.0f);
        std::vector<float> beta(d, 0.0f);
        weights.ln1_gamma[l] = alloc_and_copy(gamma.data(), gamma.size());
        weights.ln1_beta[l]  = alloc_and_copy(beta.data(), beta.size());
        weights.ln2_gamma[l] = alloc_and_copy(gamma.data(), gamma.size());
        weights.ln2_beta[l]  = alloc_and_copy(beta.data(), beta.size());

        // mlp
        std::vector<float> w1(static_cast<std::size_t>(d) * mh);
        xavier_uniform(w1.data(), d, mh, w1.size(), rng);
        weights.mlp_w1[l] = alloc_and_copy(w1.data(), w1.size());

        std::vector<float> b1(mh, 0.0f);
        weights.mlp_b1[l] = alloc_and_copy(b1.data(), b1.size());

        std::vector<float> w2(static_cast<std::size_t>(mh) * d);
        xavier_uniform(w2.data(), mh, d, w2.size(), rng);
        weights.mlp_w2[l] = alloc_and_copy(w2.data(), w2.size());

        std::vector<float> b2(d, 0.0f);
        weights.mlp_b2[l] = alloc_and_copy(b2.data(), b2.size());
    }

    // final layer norm
    {
        std::vector<float> gamma(d, 1.0f);
        std::vector<float> beta(d, 0.0f);
        weights.final_ln_gamma = alloc_and_copy(gamma.data(), gamma.size());
        weights.final_ln_beta  = alloc_and_copy(beta.data(), beta.size());
    }

    // classification head
    {
        std::vector<float> w(static_cast<std::size_t>(d) * nc);
        xavier_uniform(w.data(), d, nc, w.size(), rng);
        weights.head_weight = alloc_and_copy(w.data(), w.size());

        std::vector<float> b(nc, 0.0f);
        weights.head_bias = alloc_and_copy(b.data(), b.size());
    }
}

// ---- free weights ----

void vit_free_weights(VitWeights& weights, const VitConfig& config) {
    cudaFree(weights.pe_weight);
    cudaFree(weights.pe_bias);
    cudaFree(weights.cls_token);
    cudaFree(weights.position_embedding);

    for (int l = 0; l < config.n_layers; ++l) {
        cudaFree(weights.ln1_gamma[l]);
        cudaFree(weights.ln1_beta[l]);
        cudaFree(weights.ln2_gamma[l]);
        cudaFree(weights.ln2_beta[l]);
        cudaFree(weights.mlp_w1[l]);
        cudaFree(weights.mlp_b1[l]);
        cudaFree(weights.mlp_w2[l]);
        cudaFree(weights.mlp_b2[l]);
    }
    delete[] weights.ln1_gamma;
    delete[] weights.ln1_beta;
    delete[] weights.ln2_gamma;
    delete[] weights.ln2_beta;
    delete[] weights.mlp_w1;
    delete[] weights.mlp_b1;
    delete[] weights.mlp_w2;
    delete[] weights.mlp_b2;

    cudaFree(weights.final_ln_gamma);
    cudaFree(weights.final_ln_beta);
    cudaFree(weights.head_weight);
    cudaFree(weights.head_bias);
}

// ---- workspace ----

std::size_t vit_workspace_bytes(const VitConfig& config, int batch) {
    const std::size_t sl = config.seq_len();
    const std::size_t d  = config.d_model;
    const std::size_t mh = config.mlp_hidden;
    const std::size_t nc = config.n_classes;
    const std::size_t B  = batch;

    std::size_t token_buf = B * sl * d;
    std::size_t mlp_buf   = B * sl * mh;

    return (16 * token_buf + 4 * mlp_buf + 4 * B * d + B * nc) * sizeof(float);
}

// ---- forward ----

void vit_forward(const float* d_image, float* d_logits,
                 float* d_workspace,
                 const VitWeights& weights,
                 const VitConfig& config,
                 int batch,
                 cudaStream_t stream) {
    const int sl = config.seq_len();
    const int d  = config.d_model;
    const int mh = config.mlp_hidden;
    const int nc = config.n_classes;

    const std::size_t token_size = static_cast<std::size_t>(batch) * sl * d;
    const std::size_t mlp_size   = static_cast<std::size_t>(batch) * sl * mh;

    // asignar workspace
    float* buf0 = d_workspace;                               // embed output / current
    float* buf1 = buf0 + token_size;                          // norm output
    float* buf2 = buf1 + token_size;                          // attn output
    float* buf3 = buf2 + token_size;                          // residual
    float* buf4 = buf3 + token_size;                          // mlp output
    float* buf5 = buf4 + token_size;                          // extra temp
    float* buf_mlp_hidden = buf5 + token_size;                // mlp hidden
    float* buf_cls = buf_mlp_hidden + mlp_size;               // cls extract

    // ---- 1. Patch Embedding ----
    PatchEmbeddingShape pe_shape;
    pe_shape.batch       = batch;
    pe_shape.in_channels = config.in_channels;
    pe_shape.height      = config.image_size;
    pe_shape.width       = config.image_size;
    pe_shape.patch_size  = config.patch_size;
    pe_shape.d_model     = d;

    patch_embedding_pipeline_cuda(
        d_image,
        weights.pe_weight,
        weights.pe_bias,
        weights.cls_token,
        weights.position_embedding,
        buf0,
        pe_shape,
        stream);

    // ---- 2. Encoder blocks ----
    float* current = buf0;

    for (int l = 0; l < config.n_layers; ++l) {
        const int rows = batch * sl;

        // --- sub-block 1: x + attention(norm1(x)) ---
        LayerNormShape ln_shape{rows, d};
        layer_norm_cuda(current, weights.ln1_gamma[l], weights.ln1_beta[l],
                        buf1, ln_shape, config.eps, stream);

        AttentionShape attn_shape{batch, sl, d, config.heads};
        multi_head_self_attention_cuda(buf1, buf2, attn_shape,
                                       nullptr, nullptr, stream);

        add_cuda(current, buf2, buf3, token_size, stream);

        // --- sub-block 2: residual + mlp(norm2(residual)) ---
        layer_norm_cuda(buf3, weights.ln2_gamma[l], weights.ln2_beta[l],
                        buf1, ln_shape, config.eps, stream);

        LinearShape lin1_shape{rows, d, mh};
        linear_cuda(buf1, weights.mlp_w1[l], weights.mlp_b1[l],
                    buf_mlp_hidden, lin1_shape, stream);
        gelu_cuda(buf_mlp_hidden, buf_mlp_hidden, mlp_size, stream);

        LinearShape lin2_shape{rows, mh, d};
        linear_cuda(buf_mlp_hidden, weights.mlp_w2[l], weights.mlp_b2[l],
                    buf4, lin2_shape, stream);

        float* next = (l < config.n_layers - 1) ? buf0 : buf5;
        add_cuda(buf3, buf4, next, token_size, stream);
        current = next;
    }

    // ---- 3. Final LayerNorm ----
    {
        const int rows = batch * sl;
        LayerNormShape ln_shape{rows, d};
        layer_norm_cuda(current, weights.final_ln_gamma, weights.final_ln_beta,
                        buf1, ln_shape, config.eps, stream);
    }

    // ---- 4. Extract CLS token (token 0 de cada muestra, copia strided en una sola llamada) ----
    check_cuda(cudaMemcpy2DAsync(
        buf_cls, static_cast<std::size_t>(d) * sizeof(float),      // destino: filas contiguas de d
        buf1, static_cast<std::size_t>(sl) * d * sizeof(float),    // origen: salto sl*d entre muestras
        static_cast<std::size_t>(d) * sizeof(float), batch,        // ancho = d floats, alto = batch
        cudaMemcpyDeviceToDevice, stream), "extract CLS token");

    // ---- 5. Classification head ----
    {
        LinearShape head_shape{batch, d, nc};
        linear_cuda(buf_cls, weights.head_weight, weights.head_bias,
                    d_logits, head_shape, stream);
    }
}

// ---- gradientes y backward pass ----

void vit_init_gradients(VitGradients& grads, const VitConfig& config) {
    const int d = config.d_model;
    const int pd = config.patch_dim();
    const int sl = config.seq_len();
    const int mh = config.mlp_hidden;
    const int nc = config.n_classes;
    const int nl = config.n_layers;

    grads.pe_weight = alloc_gpu(static_cast<std::size_t>(pd) * d);
    grads.pe_bias = alloc_gpu(d);
    grads.cls_token = alloc_gpu(d);
    grads.position_embedding = alloc_gpu(static_cast<std::size_t>(sl) * d);

    grads.ln1_gamma = new float*[nl];
    grads.ln1_beta  = new float*[nl];
    grads.ln2_gamma = new float*[nl];
    grads.ln2_beta  = new float*[nl];
    grads.mlp_w1    = new float*[nl];
    grads.mlp_b1    = new float*[nl];
    grads.mlp_w2    = new float*[nl];
    grads.mlp_b2    = new float*[nl];

    for (int l = 0; l < nl; ++l) {
        grads.ln1_gamma[l] = alloc_gpu(d);
        grads.ln1_beta[l]  = alloc_gpu(d);
        grads.ln2_gamma[l] = alloc_gpu(d);
        grads.ln2_beta[l]  = alloc_gpu(d);

        grads.mlp_w1[l] = alloc_gpu(static_cast<std::size_t>(d) * mh);
        grads.mlp_b1[l] = alloc_gpu(mh);
        grads.mlp_w2[l] = alloc_gpu(static_cast<std::size_t>(mh) * d);
        grads.mlp_b2[l] = alloc_gpu(d);
    }

    grads.final_ln_gamma = alloc_gpu(d);
    grads.final_ln_beta  = alloc_gpu(d);
    grads.head_weight    = alloc_gpu(static_cast<std::size_t>(d) * nc);
    grads.head_bias      = alloc_gpu(nc);
}

void vit_zero_gradients(VitGradients& grads, const VitConfig& config, cudaStream_t stream) {
    const int d = config.d_model;
    const int pd = config.patch_dim();
    const int sl = config.seq_len();
    const int mh = config.mlp_hidden;
    const int nc = config.n_classes;
    const int nl = config.n_layers;

    check_cuda(cudaMemsetAsync(grads.pe_weight, 0, static_cast<std::size_t>(pd) * d * sizeof(float), stream), "memset pe_weight");
    check_cuda(cudaMemsetAsync(grads.pe_bias, 0, d * sizeof(float), stream), "memset pe_bias");
    check_cuda(cudaMemsetAsync(grads.cls_token, 0, d * sizeof(float), stream), "memset cls_token");
    check_cuda(cudaMemsetAsync(grads.position_embedding, 0, static_cast<std::size_t>(sl) * d * sizeof(float), stream), "memset pos_embed");

    for (int l = 0; l < nl; ++l) {
        check_cuda(cudaMemsetAsync(grads.ln1_gamma[l], 0, d * sizeof(float), stream), "memset ln1_g");
        check_cuda(cudaMemsetAsync(grads.ln1_beta[l], 0, d * sizeof(float), stream), "memset ln1_b");
        check_cuda(cudaMemsetAsync(grads.ln2_gamma[l], 0, d * sizeof(float), stream), "memset ln2_g");
        check_cuda(cudaMemsetAsync(grads.ln2_beta[l], 0, d * sizeof(float), stream), "memset ln2_b");
        check_cuda(cudaMemsetAsync(grads.mlp_w1[l], 0, static_cast<std::size_t>(d) * mh * sizeof(float), stream), "memset mlp_w1");
        check_cuda(cudaMemsetAsync(grads.mlp_b1[l], 0, mh * sizeof(float), stream), "memset mlp_b1");
        check_cuda(cudaMemsetAsync(grads.mlp_w2[l], 0, static_cast<std::size_t>(mh) * d * sizeof(float), stream), "memset mlp_w2");
        check_cuda(cudaMemsetAsync(grads.mlp_b2[l], 0, d * sizeof(float), stream), "memset mlp_b2");
    }

    check_cuda(cudaMemsetAsync(grads.final_ln_gamma, 0, d * sizeof(float), stream), "memset final_g");
    check_cuda(cudaMemsetAsync(grads.final_ln_beta, 0, d * sizeof(float), stream), "memset final_b");
    check_cuda(cudaMemsetAsync(grads.head_weight, 0, static_cast<std::size_t>(d) * nc * sizeof(float), stream), "memset head_w");
    check_cuda(cudaMemsetAsync(grads.head_bias, 0, nc * sizeof(float), stream), "memset head_b");
}

void vit_free_gradients(VitGradients& grads, const VitConfig& config) {
    vit_free_weights(grads, config);
}

void vit_update_weights_sgd(VitWeights& weights,
                            const VitGradients& grads,
                            const VitConfig& config,
                            float learning_rate,
                            cudaStream_t stream) {
    const int d = config.d_model;
    const int pd = config.patch_dim();
    const int sl = config.seq_len();
    const int mh = config.mlp_hidden;
    const int nc = config.n_classes;
    const int nl = config.n_layers;

    sgd_update_cuda(weights.pe_weight, grads.pe_weight, learning_rate, static_cast<std::size_t>(pd) * d, stream);
    sgd_update_cuda(weights.pe_bias, grads.pe_bias, learning_rate, d, stream);
    sgd_update_cuda(weights.cls_token, grads.cls_token, learning_rate, d, stream);
    sgd_update_cuda(weights.position_embedding, grads.position_embedding, learning_rate, static_cast<std::size_t>(sl) * d, stream);

    for (int l = 0; l < nl; ++l) {
        sgd_update_cuda(weights.ln1_gamma[l], grads.ln1_gamma[l], learning_rate, d, stream);
        sgd_update_cuda(weights.ln1_beta[l], grads.ln1_beta[l], learning_rate, d, stream);
        sgd_update_cuda(weights.ln2_gamma[l], grads.ln2_gamma[l], learning_rate, d, stream);
        sgd_update_cuda(weights.ln2_beta[l], grads.ln2_beta[l], learning_rate, d, stream);
        sgd_update_cuda(weights.mlp_w1[l], grads.mlp_w1[l], learning_rate, static_cast<std::size_t>(d) * mh, stream);
        sgd_update_cuda(weights.mlp_b1[l], grads.mlp_b1[l], learning_rate, mh, stream);
        sgd_update_cuda(weights.mlp_w2[l], grads.mlp_w2[l], learning_rate, static_cast<std::size_t>(mh) * d, stream);
        sgd_update_cuda(weights.mlp_b2[l], grads.mlp_b2[l], learning_rate, d, stream);
    }

    sgd_update_cuda(weights.final_ln_gamma, grads.final_ln_gamma, learning_rate, d, stream);
    sgd_update_cuda(weights.final_ln_beta, grads.final_ln_beta, learning_rate, d, stream);
    sgd_update_cuda(weights.head_weight, grads.head_weight, learning_rate, static_cast<std::size_t>(d) * nc, stream);
    sgd_update_cuda(weights.head_bias, grads.head_bias, learning_rate, nc, stream);
}

namespace {

__global__ void scatter_cls_grad_kernel(const float* grad_cls, float* grad_seq,
                                        int batch, int seq_len, int d_model) {
    int b = blockIdx.x;
    int d = threadIdx.x;
    if (b < batch && d < d_model) {
        std::size_t seq_idx = (static_cast<std::size_t>(b) * seq_len + 0) * d_model + d;
        grad_seq[seq_idx] = grad_cls[static_cast<std::size_t>(b) * d_model + d];
    }
}

}  // namespace

void vit_backward(const float* d_image,
                  const float* d_grad_logits,
                  float* d_workspace,
                  const VitWeights& weights,
                  VitGradients& grads,
                  const VitConfig& config,
                  int batch,
                  cudaStream_t stream) {
    const int sl = config.seq_len();
    const int d  = config.d_model;
    const int mh = config.mlp_hidden;
    const int nc = config.n_classes;

    const std::size_t token_size = static_cast<std::size_t>(batch) * sl * d;
    const std::size_t mlp_size   = static_cast<std::size_t>(batch) * sl * mh;

    float* buf0 = d_workspace;                               // embed output / current
    float* buf1 = buf0 + token_size;                          // norm output
    float* buf2 = buf1 + token_size;                          // attn output
    float* buf3 = buf2 + token_size;                          // residual
    float* buf4 = buf3 + token_size;                          // mlp output
    float* buf5 = buf4 + token_size;                          // extra temp
    float* buf_mlp_hidden = buf5 + token_size;                // mlp hidden
    float* buf_cls = buf_mlp_hidden + mlp_size;               // cls extract

    float* d_grad_cls = buf_cls + batch * d;                  // [batch, d]
    float* d_grad_seq = d_grad_cls + batch * d;               // [batch, sl, d]
    float* d_grad_tmp = d_grad_seq + token_size;              // [batch, sl, d]
    float* d_grad_mlp = d_grad_tmp + token_size;              // [batch, sl, mh]

    // 1. Classification head backward: cls -> logits
    LinearShape head_shape{batch, d, nc};
    linear_backward_cuda(buf_cls, weights.head_weight, d_grad_logits,
                         d_grad_cls, grads.head_weight, grads.head_bias,
                         head_shape, stream);

    // 2. Scatter d_grad_cls a d_grad_seq
    check_cuda(cudaMemsetAsync(d_grad_seq, 0, token_size * sizeof(float), stream), "zero d_grad_seq");
    scatter_cls_grad_kernel<<<batch, d, 0, stream>>>(d_grad_cls, d_grad_seq, batch, sl, d);
    check_cuda(cudaGetLastError(), "scatter_cls_grad_kernel");

    // 3. Final LayerNorm backward
    const int rows = batch * sl;
    LayerNormShape ln_shape{rows, d};
    float* current = (config.n_layers > 0) ? buf5 : buf0;
    layer_norm_backward_cuda(current, weights.final_ln_gamma, d_grad_seq, d_grad_tmp,
                            grads.final_ln_gamma, grads.final_ln_beta, ln_shape, config.eps, stream);

    // 4. Encoder layers backward (capas en orden inverso: nl-1 -> 0)
    for (int l = config.n_layers - 1; l >= 0; --l) {
        LinearShape lin2_shape{rows, mh, d};
        linear_backward_cuda(buf_mlp_hidden, weights.mlp_w2[l], d_grad_tmp,
                             d_grad_mlp, grads.mlp_w2[l], grads.mlp_b2[l], lin2_shape, stream);

        gelu_backward_cuda(buf_mlp_hidden, d_grad_mlp, d_grad_mlp, mlp_size, stream);

        LinearShape lin1_shape{rows, d, mh};
        linear_backward_cuda(buf1, weights.mlp_w1[l], d_grad_mlp,
                             d_grad_seq, grads.mlp_w1[l], grads.mlp_b1[l], lin1_shape, stream);

        layer_norm_backward_cuda(buf3, weights.ln2_gamma[l], d_grad_seq, d_grad_tmp,
                                grads.ln2_gamma[l], grads.ln2_beta[l], ln_shape, config.eps, stream);

        LinearShape attn_lin_shape{rows, d, d};
        linear_backward_cuda(buf1, weights.ln1_gamma[l], d_grad_tmp, d_grad_seq,
                             nullptr, nullptr, attn_lin_shape, stream);

        layer_norm_backward_cuda(current, weights.ln1_gamma[l], d_grad_seq, d_grad_tmp,
                                grads.ln1_gamma[l], grads.ln1_beta[l], ln_shape, config.eps, stream);
    }

    // 5. Patch Embedding backward
    PatchEmbeddingShape pe_shape;
    pe_shape.batch       = batch;
    pe_shape.in_channels = config.in_channels;
    pe_shape.height      = config.image_size;
    pe_shape.width       = config.image_size;
    pe_shape.patch_size  = config.patch_size;
    pe_shape.d_model     = d;

    patch_embedding_pipeline_backward_cuda(
        d_image,
        d_grad_tmp,
        grads.pe_weight,
        grads.pe_bias,
        grads.cls_token,
        grads.position_embedding,
        pe_shape,
        stream);
}

}  // namespace vit
