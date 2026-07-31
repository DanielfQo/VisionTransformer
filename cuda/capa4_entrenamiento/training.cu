#include "training.cuh"
#include "mnist_loader.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

namespace vit {

// ---- cross-entropy loss kernel ----
// cada hilo procesa un sample del batch
// loss_per_sample[b] = -log(softmax(logits[b])[label[b]])

namespace {

__global__ void cross_entropy_kernel(const float* logits, const int* labels,
                                      float* loss_per_sample,
                                      int batch, int n_classes) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) {
        return;
    }

    const float* row = logits + static_cast<std::size_t>(b) * n_classes;
    int label = labels[b];

    // log-softmax con estabilidad numerica
    float max_val = row[0];
    for (int c = 1; c < n_classes; ++c) {
        max_val = fmaxf(max_val, row[c]);
    }

    float sum_exp = 0.0f;
    for (int c = 0; c < n_classes; ++c) {
        sum_exp += expf(row[c] - max_val);
    }

    float log_softmax = (row[label] - max_val) - logf(sum_exp);
    loss_per_sample[b] = -log_softmax;
}

// kernel para reducir la suma del loss en la GPU
__global__ void reduce_sum_kernel(const float* input, float* output, int n) {
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        local_sum += input[i];
    }
    float total = block_reduce_sum(local_sum);
    if (threadIdx.x == 0) {
        *output = total;
    }
}

__global__ void sgd_kernel(float* param, const float* grad,
                           float lr, std::size_t n) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        param[i] -= lr * grad[i];
    }
}

__global__ void adam_kernel(float* param, const float* grad,
                            float* m, float* v,
                            float lr, float beta1, float beta2, float eps,
                            float weight_decay,
                            float bias_correction1, float bias_correction2,
                            std::size_t n) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    float g = grad[i];

    // momentos exponenciales
    float m_i = beta1 * m[i] + (1.0f - beta1) * g;
    float v_i = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = m_i;
    v[i] = v_i;

    // correccion de sesgo: m y v arrancan en 0 y estarian subestimados sin esto
    float m_hat = m_i / bias_correction1;
    float v_hat = v_i / bias_correction2;

    // weight decay desacoplado (AdamW): no pasa por los momentos
    param[i] -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param[i]);
}

__global__ void cross_entropy_backward_kernel(const float* logits, const int* labels,
                                              float* grad_logits, int batch, int n_classes) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) {
        return;
    }

    const float* row = logits + static_cast<std::size_t>(b) * n_classes;
    float* grad_row = grad_logits + static_cast<std::size_t>(b) * n_classes;
    int label = labels[b];

    float max_val = row[0];
    for (int c = 1; c < n_classes; ++c) {
        max_val = fmaxf(max_val, row[c]);
    }

    float sum_exp = 0.0f;
    for (int c = 0; c < n_classes; ++c) {
        sum_exp += expf(row[c] - max_val);
    }

    float inv_batch = 1.0f / static_cast<float>(batch);
    for (int c = 0; c < n_classes; ++c) {
        float p = expf(row[c] - max_val) / sum_exp;
        float target = (c == label) ? 1.0f : 0.0f;
        grad_row[c] = (p - target) * inv_batch;
    }
}

}  // namespace

float cross_entropy_loss(const float* d_logits, const int* d_labels,
                         int batch, int n_classes,
                         float* d_loss_workspace,
                         cudaStream_t stream) {
    int blocks = (batch + kBlockSize - 1) / kBlockSize;
    cross_entropy_kernel<<<blocks, kBlockSize, 0, stream>>>(
        d_logits, d_labels, d_loss_workspace, batch, n_classes);
    check_cuda(cudaGetLastError(), "cross_entropy_kernel");

    // reduccion paralela del loss directamente en la GPU
    float* d_total_sum = d_loss_workspace + batch;
    reduce_sum_kernel<<<1, kBlockSize, 0, stream>>>(d_loss_workspace, d_total_sum, batch);
    check_cuda(cudaGetLastError(), "reduce_sum_kernel");

    float h_total_loss = 0.0f;
    check_cuda(cudaMemcpyAsync(&h_total_loss, d_total_sum,
                                sizeof(float),
                                cudaMemcpyDeviceToHost, stream),
               "copy total loss to host");
    check_cuda(cudaStreamSynchronize(stream), "sync loss");

    return h_total_loss / static_cast<float>(batch);
}

void cross_entropy_backward_cuda(const float* d_logits, const int* d_labels,
                                 float* d_grad_logits, int batch, int n_classes,
                                 cudaStream_t stream) {
    int blocks = (batch + kBlockSize - 1) / kBlockSize;
    cross_entropy_backward_kernel<<<blocks, kBlockSize, 0, stream>>>(
        d_logits, d_labels, d_grad_logits, batch, n_classes);
    check_cuda(cudaGetLastError(), "cross_entropy_backward_kernel");
}

void sgd_update_cuda(float* d_param, const float* d_grad,
                     float lr, std::size_t n,
                     cudaStream_t stream) {
    std::size_t blocks = (n + kBlockSize - 1) / kBlockSize;
    sgd_kernel<<<blocks, kBlockSize, 0, stream>>>(d_param, d_grad, lr, n);
    check_cuda(cudaGetLastError(), "sgd_kernel");
}

// ---- optimizador Adam ----

void adam_init(AdamState& state, const VitConfig& config) {
    // m y v tienen la forma de los pesos: reutilizamos el allocador de gradientes
    vit_init_gradients(state.m, config);
    vit_init_gradients(state.v, config);
    vit_zero_gradients(state.m, config, nullptr);
    vit_zero_gradients(state.v, config, nullptr);
    state.step = 0;
    check_cuda(cudaDeviceSynchronize(), "adam_init zero");
}

void adam_free(AdamState& state, const VitConfig& config) {
    vit_free_gradients(state.m, config);
    vit_free_gradients(state.v, config);
    state.step = 0;
}

void adam_update_cuda(float* d_param, const float* d_grad,
                      float* d_m, float* d_v,
                      float lr, const AdamConfig& cfg,
                      float bias_correction1, float bias_correction2,
                      std::size_t n, cudaStream_t stream) {
    std::size_t blocks = (n + kBlockSize - 1) / kBlockSize;
    adam_kernel<<<blocks, kBlockSize, 0, stream>>>(
        d_param, d_grad, d_m, d_v,
        lr, cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay,
        bias_correction1, bias_correction2, n);
    check_cuda(cudaGetLastError(), "adam_kernel");
}

void vit_update_weights_adam(VitWeights& weights, const VitGradients& grads,
                             AdamState& state, const VitConfig& config,
                             float lr, const AdamConfig& cfg,
                             cudaStream_t stream) {
    ++state.step;

    // 1 - beta^t, con t = numero de pasos ya aplicados
    const float t = static_cast<float>(state.step);
    const float bc1 = 1.0f - std::pow(cfg.beta1, t);
    const float bc2 = 1.0f - std::pow(cfg.beta2, t);

    // las cuatro listas se corresponden elemento a elemento porque salen
    // del mismo tipo de estructura y del mismo orden de enumeracion
    const std::vector<TensorRef> params = vit_tensor_list(weights, config);
    const std::vector<TensorRef> gradients = vit_tensor_list(grads, config);
    const std::vector<TensorRef> m_state = vit_tensor_list(state.m, config);
    const std::vector<TensorRef> v_state = vit_tensor_list(state.v, config);

    for (std::size_t i = 0; i < params.size(); ++i) {
        adam_update_cuda(params[i].data, gradients[i].data,
                         m_state[i].data, v_state[i].data,
                         lr, cfg, bc1, bc2, params[i].size, stream);
    }
}

// ---- scheduler de learning rate ----

float lr_schedule(long long step, long long warmup_steps, long long total_steps,
                  float base_lr, float min_lr) {
    if (warmup_steps > 0 && step < warmup_steps) {
        // rampa lineal desde base_lr/warmup_steps hasta base_lr
        return base_lr * static_cast<float>(step + 1) / static_cast<float>(warmup_steps);
    }

    long long decay_steps = total_steps - warmup_steps;
    if (decay_steps <= 0) {
        return base_lr;
    }

    float progress = static_cast<float>(step - warmup_steps) / static_cast<float>(decay_steps);
    progress = std::min(1.0f, std::max(0.0f, progress));

    const float pi = 3.14159265358979323846f;
    return min_lr + 0.5f * (base_lr - min_lr) * (1.0f + std::cos(pi * progress));
}

// ---- training loop ----

float train_vit(const float* h_train_images, const int* h_train_labels, int n_train,
                const float* h_val_images, const int* h_val_labels, int n_val,
                const float* h_test_images, const int* h_test_labels, int n_test,
                VitWeights& weights, const VitConfig& config,
                const TrainConfig& train_cfg) {
    const int img_size = config.in_channels * config.image_size * config.image_size;
    const int nc = config.n_classes;

    const int epochs = train_cfg.epochs;
    const int batch_size = train_cfg.batch_size;

    // allocate GPU buffers
    float* d_images = nullptr;
    float* d_logits = nullptr;
    float* d_grad_logits = nullptr;
    int*   d_labels = nullptr;
    float* d_loss_ws = nullptr;

    std::size_t workspace_bytes = vit_workspace_bytes(config, batch_size);
    float* d_workspace = nullptr;

    check_cuda(cudaMalloc(&d_images, static_cast<std::size_t>(batch_size) * img_size * sizeof(float)),
               "alloc d_images");
    check_cuda(cudaMalloc(&d_logits, static_cast<std::size_t>(batch_size) * nc * sizeof(float)),
               "alloc d_logits");
    check_cuda(cudaMalloc(&d_grad_logits, static_cast<std::size_t>(batch_size) * nc * sizeof(float)),
               "alloc d_grad_logits");
    check_cuda(cudaMalloc(&d_labels, static_cast<std::size_t>(batch_size) * sizeof(int)),
               "alloc d_labels");
    check_cuda(cudaMalloc(&d_loss_ws, (static_cast<std::size_t>(batch_size) + 1) * sizeof(float)),
               "alloc d_loss_ws");
    check_cuda(cudaMalloc(&d_workspace, workspace_bytes), "alloc workspace");

    VitGradients grads;
    vit_init_gradients(grads, config);

    AdamState adam_state;
    if (train_cfg.use_adam) {
        adam_init(adam_state, config);
    }

    // indices para shuffle de entrenamiento
    std::vector<int> indices(n_train);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(123);

    int n_batches = n_train / batch_size;
    int n_val_batches = n_val / batch_size;

    // el scheduler avanza por paso (batch), no por epoca
    const long long total_steps = static_cast<long long>(epochs) * n_batches;
    const long long warmup_steps =
        static_cast<long long>(train_cfg.warmup_ratio * static_cast<float>(total_steps));
    long long global_step = 0;
    float current_lr = train_cfg.lr;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), rng);

        float epoch_train_loss = 0.0f;
        int train_loss_count = 0;

        for (int b = 0; b < n_batches; ++b) {
            // preparar batch en host
            std::vector<float> h_batch_img(static_cast<std::size_t>(batch_size) * img_size);
            std::vector<int>   h_batch_lbl(batch_size);

            for (int i = 0; i < batch_size; ++i) {
                int idx = indices[b * batch_size + i];
                std::copy(h_train_images + static_cast<std::size_t>(idx) * img_size,
                          h_train_images + static_cast<std::size_t>(idx + 1) * img_size,
                          h_batch_img.data() + static_cast<std::size_t>(i) * img_size);
                h_batch_lbl[i] = h_train_labels[idx];
            }

            // copiar a GPU de forma asincrona
            check_cuda(cudaMemcpyAsync(d_images, h_batch_img.data(),
                                       h_batch_img.size() * sizeof(float),
                                       cudaMemcpyHostToDevice, nullptr), "copy images H->D");
            check_cuda(cudaMemcpyAsync(d_labels, h_batch_lbl.data(),
                                       h_batch_lbl.size() * sizeof(int),
                                       cudaMemcpyHostToDevice, nullptr), "copy labels H->D");

            // reiniciar gradientes
            vit_zero_gradients(grads, config, nullptr);

            // forward pass
            vit_forward(d_images, d_logits, d_workspace, weights, config,
                        batch_size, nullptr);

            // loss
            float loss = cross_entropy_loss(d_logits, d_labels, batch_size,
                                             nc, d_loss_ws, nullptr);
            epoch_train_loss += loss;
            ++train_loss_count;

            // backward pass
            cross_entropy_backward_cuda(d_logits, d_labels, d_grad_logits,
                                        batch_size, nc, nullptr);
            vit_backward(d_images, d_grad_logits, d_workspace, weights,
                         grads, config, batch_size, nullptr);

            // learning rate de este paso segun el scheduler
            current_lr = train_cfg.use_adam
                ? lr_schedule(global_step, warmup_steps, total_steps,
                              train_cfg.lr, train_cfg.min_lr)
                : train_cfg.lr;

            // actualizar pesos
            if (train_cfg.use_adam) {
                vit_update_weights_adam(weights, grads, adam_state, config,
                                        current_lr, train_cfg.adam, nullptr);
            } else {
                vit_update_weights_sgd(weights, grads, config, current_lr, nullptr);
            }

            ++global_step;
        }

        float avg_train_loss = train_loss_count > 0 ? epoch_train_loss / train_loss_count : 0.0f;

        // ---- evaluacion de validacion por epoca ----
        float epoch_val_loss = 0.0f;
        int val_loss_count = 0;
        int val_correct = 0;

        for (int b = 0; b < n_val_batches; ++b) {
            std::vector<float> h_batch_img(static_cast<std::size_t>(batch_size) * img_size);
            std::vector<int>   h_batch_lbl(batch_size);

            for (int i = 0; i < batch_size; ++i) {
                int idx = b * batch_size + i;
                std::copy(h_val_images + static_cast<std::size_t>(idx) * img_size,
                          h_val_images + static_cast<std::size_t>(idx + 1) * img_size,
                          h_batch_img.data() + static_cast<std::size_t>(i) * img_size);
                h_batch_lbl[i] = h_val_labels[idx];
            }

            check_cuda(cudaMemcpyAsync(d_images, h_batch_img.data(),
                                       h_batch_img.size() * sizeof(float),
                                       cudaMemcpyHostToDevice, nullptr), "copy val images");
            check_cuda(cudaMemcpyAsync(d_labels, h_batch_lbl.data(),
                                       h_batch_lbl.size() * sizeof(int),
                                       cudaMemcpyHostToDevice, nullptr), "copy val labels");

            vit_forward(d_images, d_logits, d_workspace, weights, config,
                        batch_size, nullptr);

            float v_loss = cross_entropy_loss(d_logits, d_labels, batch_size,
                                              nc, d_loss_ws, nullptr);
            epoch_val_loss += v_loss;
            ++val_loss_count;

            // calcular accuracy de validacion
            std::vector<float> h_logits(static_cast<std::size_t>(batch_size) * nc);
            check_cuda(cudaMemcpy(h_logits.data(), d_logits,
                                  h_logits.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost), "copy val logits D->H");

            for (int i = 0; i < batch_size; ++i) {
                int pred = 0;
                float max_val = h_logits[static_cast<std::size_t>(i) * nc];
                for (int c = 1; c < nc; ++c) {
                    float val = h_logits[static_cast<std::size_t>(i) * nc + c];
                    if (val > max_val) {
                        max_val = val;
                        pred = c;
                    }
                }
                if (pred == h_batch_lbl[i]) {
                    ++val_correct;
                }
            }
        }

        float avg_val_loss = val_loss_count > 0 ? epoch_val_loss / val_loss_count : 0.0f;
        float val_acc = static_cast<float>(val_correct) / static_cast<float>(n_val_batches * batch_size);

        std::printf("Epoch %d/%d -- lr: %.2e -- train_loss: %.4f -- val_loss: %.4f -- val_acc: %.2f%%\n",
                    epoch + 1, epochs, current_lr, avg_train_loss, avg_val_loss, val_acc * 100.0f);
        std::fflush(stdout);
    }

    // ---- evaluacion final en test ----
    int correct = 0;
    int n_test_batches = n_test / batch_size;

    for (int b = 0; b < n_test_batches; ++b) {
        std::vector<float> h_batch_img(static_cast<std::size_t>(batch_size) * img_size);
        std::vector<int>   h_batch_lbl(batch_size);

        for (int i = 0; i < batch_size; ++i) {
            int idx = b * batch_size + i;
            std::copy(h_test_images + static_cast<std::size_t>(idx) * img_size,
                      h_test_images + static_cast<std::size_t>(idx + 1) * img_size,
                      h_batch_img.data() + static_cast<std::size_t>(i) * img_size);
            h_batch_lbl[i] = h_test_labels[idx];
        }

        check_cuda(cudaMemcpy(d_images, h_batch_img.data(),
                              h_batch_img.size() * sizeof(float),
                              cudaMemcpyHostToDevice), "copy test images");

        vit_forward(d_images, d_logits, d_workspace, weights, config,
                    batch_size, nullptr);

        // copiar logits a CPU y calcular accuracy
        std::vector<float> h_logits(static_cast<std::size_t>(batch_size) * nc);
        check_cuda(cudaMemcpy(h_logits.data(), d_logits,
                              h_logits.size() * sizeof(float),
                              cudaMemcpyDeviceToHost), "copy logits D->H");

        for (int i = 0; i < batch_size; ++i) {
            int pred = 0;
            float max_val = h_logits[static_cast<std::size_t>(i) * nc];
            for (int c = 1; c < nc; ++c) {
                float val = h_logits[static_cast<std::size_t>(i) * nc + c];
                if (val > max_val) {
                    max_val = val;
                    pred = c;
                }
            }
            if (pred == h_batch_lbl[i]) {
                ++correct;
            }
        }
    }

    float accuracy = static_cast<float>(correct) /
                     static_cast<float>(n_test_batches * batch_size);
    std::printf("Test accuracy: %.2f%% (%d/%d)\n",
                accuracy * 100.0f, correct, n_test_batches * batch_size);

    // cleanup
    if (train_cfg.use_adam) {
        adam_free(adam_state, config);
    }
    vit_free_gradients(grads, config);
    cudaFree(d_images);
    cudaFree(d_logits);
    cudaFree(d_grad_logits);
    cudaFree(d_labels);
    cudaFree(d_loss_ws);
    cudaFree(d_workspace);

    return accuracy;
}

}  // namespace vit
