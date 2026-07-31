#include "vit.cuh"

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

    // buffers necesarios (pueden reutilizarse, necesitamos el maximo):
    // buf_embed:    B * sl * d     (salida del patch embedding)
    // buf_norm:     B * sl * d     (salida de layer norm)
    // buf_attn:     B * sl * d     (salida de attention)
    // buf_residual: B * sl * d     (residual temporal)
    // buf_mlp_h:    B * sl * mh    (hidden del MLP)
    // buf_mlp_out:  B * sl * d     (salida del MLP)
    // buf_cls:      B * d          (CLS token extraido)
    // buf_logits:   B * nc         (logits finales)
    // Total: necesitamos 6 buffers de tamano B*sl*d + 1 de B*sl*mh + B*d + B*nc

    std::size_t token_buf = B * sl * d;
    std::size_t mlp_buf   = B * sl * mh;

    // usamos 6 token buffers + 1 mlp hidden buffer
    return (6 * token_buf + mlp_buf + B * d + B * nc) * sizeof(float);
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
    // d_logits se usa directamente para la salida

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
    // current = buf0
    float* current = buf0;

    for (int l = 0; l < config.n_layers; ++l) {
        const int rows = batch * sl;

        // --- sub-block 1: x + attention(norm1(x)) ---
        // norm1
        LayerNormShape ln_shape{rows, d};
        layer_norm_cuda(current, weights.ln1_gamma[l], weights.ln1_beta[l],
                        buf1, ln_shape, 1e-5f, stream);

        // attention
        AttentionShape attn_shape{batch, sl, d, config.heads};
        multi_head_self_attention_cuda(buf1, buf2, attn_shape,
                                       nullptr, nullptr, stream);

        // residual: buf3 = current + buf2
        add_cuda(current, buf2, buf3, token_size, stream);

        // --- sub-block 2: residual + mlp(norm2(residual)) ---
        // norm2
        layer_norm_cuda(buf3, weights.ln2_gamma[l], weights.ln2_beta[l],
                        buf1, ln_shape, 1e-5f, stream);

        // mlp: linear1 -> gelu -> linear2
        LinearShape lin1_shape{rows, d, mh};
        linear_cuda(buf1, weights.mlp_w1[l], weights.mlp_b1[l],
                    buf_mlp_hidden, lin1_shape, stream);
        gelu_cuda(buf_mlp_hidden, buf_mlp_hidden, mlp_size, stream);

        LinearShape lin2_shape{rows, mh, d};
        linear_cuda(buf_mlp_hidden, weights.mlp_w2[l], weights.mlp_b2[l],
                    buf4, lin2_shape, stream);

        // residual: current = buf3 + buf4
        // Si no es la ultima capa, result va a buf0 (current); si es la ultima a buf5
        float* next = (l < config.n_layers - 1) ? buf0 : buf5;
        add_cuda(buf3, buf4, next, token_size, stream);
        current = next;
    }

    // ---- 3. Final LayerNorm ----
    {
        const int rows = batch * sl;
        LayerNormShape ln_shape{rows, d};
        layer_norm_cuda(current, weights.final_ln_gamma, weights.final_ln_beta,
                        buf1, ln_shape, 1e-5f, stream);
    }

    // ---- 4. Extract CLS token (token 0 de cada batch) ----
    // buf1 contiene [batch, seq_len, d_model], extraemos token 0
    // Usamos cudaMemcpy2D: copiar d floats con stride sl*d
    for (int b = 0; b < batch; ++b) {
        check_cuda(cudaMemcpyAsync(
            buf_cls + static_cast<std::size_t>(b) * d,
            buf1 + static_cast<std::size_t>(b) * sl * d,
            d * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream), "extract CLS token");
    }

    // ---- 5. Classification head: linear(cls) -> logits ----
    {
        LinearShape head_shape{batch, d, nc};
        linear_cuda(buf_cls, weights.head_weight, weights.head_bias,
                    d_logits, head_shape, stream);
    }
}

}  // namespace vit
