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
                                      int n_classes) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;

    const float* row = logits + static_cast<std::size_t>(b) * n_classes;
    int label = labels[b];

    // numerically stable log-softmax
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

__global__ void sgd_kernel(float* param, const float* grad,
                           float lr, std::size_t n) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        param[i] -= lr * grad[i];
    }
}

}  // namespace

float cross_entropy_loss(const float* d_logits, const int* d_labels,
                         int batch, int n_classes,
                         float* d_loss_workspace,
                         cudaStream_t stream) {
    int blocks = (batch + kBlockSize - 1) / kBlockSize;
    cross_entropy_kernel<<<blocks, kBlockSize, 0, stream>>>(
        d_logits, d_labels, d_loss_workspace, n_classes);
    check_cuda(cudaGetLastError(), "cross_entropy_kernel");

    // copiar losses a CPU y promediar
    std::vector<float> h_losses(batch);
    check_cuda(cudaMemcpyAsync(h_losses.data(), d_loss_workspace,
                                batch * sizeof(float),
                                cudaMemcpyDeviceToHost, stream),
               "copy losses to host");
    check_cuda(cudaStreamSynchronize(stream), "sync losses");

    float total = 0.0f;
    for (int i = 0; i < batch; ++i) {
        total += h_losses[i];
    }
    return total / static_cast<float>(batch);
}

void sgd_update_cuda(float* d_param, const float* d_grad,
                     float lr, std::size_t n,
                     cudaStream_t stream) {
    std::size_t blocks = (n + kBlockSize - 1) / kBlockSize;
    sgd_kernel<<<blocks, kBlockSize, 0, stream>>>(d_param, d_grad, lr, n);
    check_cuda(cudaGetLastError(), "sgd_kernel");
}

// ---- training loop ----

float train_vit(const float* h_train_images, const int* h_train_labels, int n_train,
                const float* h_test_images, const int* h_test_labels, int n_test,
                VitWeights& weights, const VitConfig& config,
                int epochs, int batch_size, float learning_rate) {
    const int img_size = config.in_channels * config.image_size * config.image_size;
    const int nc = config.n_classes;

    // allocate GPU buffers
    float* d_images = nullptr;
    float* d_logits = nullptr;
    int*   d_labels = nullptr;
    float* d_loss_ws = nullptr;

    std::size_t workspace_bytes = vit_workspace_bytes(config, batch_size);
    float* d_workspace = nullptr;

    check_cuda(cudaMalloc(&d_images, static_cast<std::size_t>(batch_size) * img_size * sizeof(float)),
               "alloc d_images");
    check_cuda(cudaMalloc(&d_logits, static_cast<std::size_t>(batch_size) * nc * sizeof(float)),
               "alloc d_logits");
    check_cuda(cudaMalloc(&d_labels, static_cast<std::size_t>(batch_size) * sizeof(int)),
               "alloc d_labels");
    check_cuda(cudaMalloc(&d_loss_ws, static_cast<std::size_t>(batch_size) * sizeof(float)),
               "alloc d_loss_ws");
    check_cuda(cudaMalloc(&d_workspace, workspace_bytes), "alloc workspace");

    // indices para shuffle
    std::vector<int> indices(n_train);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(123);

    int n_batches = n_train / batch_size;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), rng);

        float epoch_loss = 0.0f;
        int loss_count = 0;

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

            // copiar a GPU
            check_cuda(cudaMemcpy(d_images, h_batch_img.data(),
                                  h_batch_img.size() * sizeof(float),
                                  cudaMemcpyHostToDevice), "copy images H->D");
            check_cuda(cudaMemcpy(d_labels, h_batch_lbl.data(),
                                  h_batch_lbl.size() * sizeof(int),
                                  cudaMemcpyHostToDevice), "copy labels H->D");

            // forward
            vit_forward(d_images, d_logits, d_workspace, weights, config,
                        batch_size, nullptr);

            // loss (sin backprop, solo para monitoreo)
            float loss = cross_entropy_loss(d_logits, d_labels, batch_size,
                                             nc, d_loss_ws, nullptr);
            epoch_loss += loss;
            ++loss_count;

            // nota: sin backward pass, los pesos no se actualizan
            // el gradiente requiere autograd que esta fuera del alcance de este proyecto
            // por ahora solo hacemos forward inference para validar la arquitectura
        }

        float avg_loss = loss_count > 0 ? epoch_loss / loss_count : 0.0f;
        std::printf("Epoch %d/%d — loss: %.4f\n", epoch + 1, epochs, avg_loss);
    }

    // ---- evaluacion en test ----
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
    cudaFree(d_images);
    cudaFree(d_logits);
    cudaFree(d_labels);
    cudaFree(d_loss_ws);
    cudaFree(d_workspace);

    return accuracy;
}

}  // namespace vit
