// Verificacion numerica del backward completo del ViT.
//
// Para una muestra de elementos de CADA tensor de pesos compara el gradiente
// analitico (vit_backward) contra diferencias finitas centradas calculadas
// con el propio forward de GPU:
//
//     grad_numerico = (L(w + h) - L(w - h)) / (2h)
//
// Si algun componente del backward esta mal (atencion, residuales, LayerNorm,
// activaciones obsoletas, etc.) este test lo detecta senalando el tensor exacto.

#include "../capa3_modelo/vit.cuh"
#include "../capa4_entrenamiento/training.cuh"

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {

float eval_loss(const float* d_image, const int* d_labels,
                const vit::VitWeights& weights, const vit::VitConfig& config,
                int batch, float* d_workspace, float* d_logits, float* d_loss_ws) {
    vit::vit_forward(d_image, d_logits, d_workspace, weights, config, batch, nullptr);
    return vit::cross_entropy_loss(d_logits, d_labels, batch, config.n_classes,
                                   d_loss_ws, nullptr);
}

// nombres en el mismo orden que vit_tensor_list
std::vector<std::string> tensor_names(int n_layers) {
    std::vector<std::string> names = {
        "pe_weight", "pe_bias", "cls_token", "position_embedding"};
    for (int l = 0; l < n_layers; ++l) {
        const std::string p = "capa" + std::to_string(l) + ".";
        names.push_back(p + "ln1_gamma");
        names.push_back(p + "ln1_beta");
        names.push_back(p + "ln2_gamma");
        names.push_back(p + "ln2_beta");
        names.push_back(p + "mlp_w1");
        names.push_back(p + "mlp_b1");
        names.push_back(p + "mlp_w2");
        names.push_back(p + "mlp_b2");
    }
    names.push_back("final_ln_gamma");
    names.push_back("final_ln_beta");
    names.push_back("head_weight");
    names.push_back("head_bias");
    return names;
}

}  // namespace

int main() {
    std::printf("=== Test: Gradientes numericos (backward completo) ===\n");

    // modelo pequeno pero con todas las piezas: 2 capas, 2 heads, CLS + 4 parches
    vit::VitConfig config;
    config.image_size  = 6;
    config.patch_size  = 3;
    config.in_channels = 1;
    config.d_model     = 8;
    config.heads       = 2;
    config.n_layers    = 2;
    config.mlp_hidden  = 16;
    config.n_classes   = 4;

    const int batch = 3;
    const int img_size = config.in_channels * config.image_size * config.image_size;
    const int nc = config.n_classes;

    // datos aleatorios reproducibles
    std::mt19937 rng(2024);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    std::vector<float> h_image(static_cast<std::size_t>(batch) * img_size);
    for (auto& v : h_image) {
        v = uni(rng);
    }
    std::vector<int> h_labels = {0, 2, 3};

    float* d_image = nullptr;
    int* d_labels = nullptr;
    vit::check_cuda(cudaMalloc(&d_image, h_image.size() * sizeof(float)), "alloc image");
    vit::check_cuda(cudaMalloc(&d_labels, h_labels.size() * sizeof(int)), "alloc labels");
    vit::check_cuda(cudaMemcpy(d_image, h_image.data(),
                               h_image.size() * sizeof(float),
                               cudaMemcpyHostToDevice), "copy image");
    vit::check_cuda(cudaMemcpy(d_labels, h_labels.data(),
                               h_labels.size() * sizeof(int),
                               cudaMemcpyHostToDevice), "copy labels");

    vit::VitWeights weights;
    vit::vit_init_weights(weights, config, 7);

    vit::VitGradients grads;
    vit::vit_init_gradients(grads, config);

    float* d_ws = nullptr;
    float* d_logits = nullptr;
    float* d_grad_logits = nullptr;
    float* d_loss_ws = nullptr;
    vit::check_cuda(cudaMalloc(&d_ws, vit::vit_workspace_bytes(config, batch)), "alloc ws");
    vit::check_cuda(cudaMalloc(&d_logits, static_cast<std::size_t>(batch) * nc * sizeof(float)), "alloc logits");
    vit::check_cuda(cudaMalloc(&d_grad_logits, static_cast<std::size_t>(batch) * nc * sizeof(float)), "alloc grad logits");
    vit::check_cuda(cudaMalloc(&d_loss_ws, (static_cast<std::size_t>(batch) + 1) * sizeof(float)), "alloc loss ws");

    // ---- gradiente analitico ----
    vit::vit_zero_gradients(grads, config, nullptr);
    float base_loss = eval_loss(d_image, d_labels, weights, config, batch,
                                d_ws, d_logits, d_loss_ws);
    vit::cross_entropy_backward_cuda(d_logits, d_labels, d_grad_logits, batch, nc, nullptr);
    vit::vit_backward(d_image, d_grad_logits, d_ws, weights, grads, config, batch, nullptr);
    vit::check_cuda(cudaDeviceSynchronize(), "sync backward");

    std::printf("  loss inicial: %.4f (azar = ln(%d) = %.4f)\n\n",
                base_loss, nc, std::log(static_cast<float>(nc)));

    const std::vector<vit::TensorRef> params = vit::vit_tensor_list(weights, config);
    const std::vector<vit::TensorRef> grad_list = vit::vit_tensor_list(grads, config);
    const std::vector<std::string> names = tensor_names(config.n_layers);

    const float h = 1e-2f;
    int total_checked = 0;
    int total_failed = 0;

    for (std::size_t k = 0; k < params.size(); ++k) {
        const std::size_t n = params[k].size;
        const std::size_t idxs[3] = {0, n / 2, n - 1};
        float max_err = 0.0f;
        bool tensor_ok = true;

        for (int s = 0; s < 3; ++s) {
            const std::size_t idx = idxs[s];
            // evitar repetir indices en tensores pequenos
            if (s == 1 && idx == idxs[0]) continue;
            if (s == 2 && (idx == idxs[0] || idx == idxs[1])) continue;

            float original = 0.0f;
            float analytic = 0.0f;
            vit::check_cuda(cudaMemcpy(&original, params[k].data + idx, sizeof(float),
                                       cudaMemcpyDeviceToHost), "read param");
            vit::check_cuda(cudaMemcpy(&analytic, grad_list[k].data + idx, sizeof(float),
                                       cudaMemcpyDeviceToHost), "read grad");

            const float wp = original + h;
            vit::check_cuda(cudaMemcpy(params[k].data + idx, &wp, sizeof(float),
                                       cudaMemcpyHostToDevice), "write w+h");
            const float loss_p = eval_loss(d_image, d_labels, weights, config, batch,
                                           d_ws, d_logits, d_loss_ws);

            const float wm = original - h;
            vit::check_cuda(cudaMemcpy(params[k].data + idx, &wm, sizeof(float),
                                       cudaMemcpyHostToDevice), "write w-h");
            const float loss_m = eval_loss(d_image, d_labels, weights, config, batch,
                                           d_ws, d_logits, d_loss_ws);

            vit::check_cuda(cudaMemcpy(params[k].data + idx, &original, sizeof(float),
                                       cudaMemcpyHostToDevice), "restore w");

            const float numeric = (loss_p - loss_m) / (2.0f * h);
            const float err = std::fabs(analytic - numeric);
            const float tol =
                0.05f * std::fmax(std::fabs(analytic), std::fabs(numeric)) + 3e-3f;

            ++total_checked;
            max_err = std::fmax(max_err, err);
            if (err > tol) {
                ++total_failed;
                tensor_ok = false;
                std::printf("  FALLO %s[%zu]: analitico=%+.6f numerico=%+.6f\n",
                            names[k].c_str(), idx, analytic, numeric);
            }
        }

        std::printf("  %-22s max|err| = %.5f ... %s\n",
                    names[k].c_str(), max_err, tensor_ok ? "OK" : "FALLO");
    }

    std::printf("\nResultado: %d/%d gradientes correctos\n",
                total_checked - total_failed, total_checked);

    // cleanup
    vit::vit_free_gradients(grads, config);
    vit::vit_free_weights(weights, config);
    cudaFree(d_image);
    cudaFree(d_labels);
    cudaFree(d_ws);
    cudaFree(d_logits);
    cudaFree(d_grad_logits);
    cudaFree(d_loss_ws);

    return total_failed == 0 ? 0 : 1;
}
