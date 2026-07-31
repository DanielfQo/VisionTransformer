#include "../capa2_componentes/patch_embedding.cuh"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace vit;

bool float_close(float a, float b, float rtol = 1e-3f, float atol = 1e-5f) {
    return std::fabs(a - b) <= atol + rtol * std::fabs(b);
}

// test: pipeline CUDA vs CPU
bool test_patch_embedding_cuda_vs_cpu() {
    std::printf("  test_patch_embedding_cuda_vs_cpu ... ");

    PatchEmbeddingShape shape;
    shape.batch       = 1;
    shape.in_channels = 1;
    shape.height      = 28;
    shape.width       = 28;
    shape.patch_size  = 7;
    shape.d_model     = 16;

    // datos de prueba
    std::vector<float> h_image(shape.image_elements());
    std::vector<float> h_weight(shape.weight_elements());
    std::vector<float> h_bias(shape.d_model, 0.1f);
    std::vector<float> h_cls(shape.d_model, 0.0f);
    std::vector<float> h_pos(shape.position_elements());

    for (std::size_t i = 0; i < h_image.size(); ++i) {
        h_image[i] = std::sin(static_cast<float>(i) * 0.01f);
    }
    for (std::size_t i = 0; i < h_weight.size(); ++i) {
        h_weight[i] = std::cos(static_cast<float>(i) * 0.02f) * 0.1f;
    }
    for (std::size_t i = 0; i < h_pos.size(); ++i) {
        h_pos[i] = static_cast<float>(i) * 0.001f;
    }

    // CPU
    std::vector<float> h_out_cpu = patch_embedding_pipeline_cpu(
        h_image, h_weight, h_bias, h_cls, h_pos, shape);

    // GPU
    float *d_img, *d_w, *d_b, *d_cls, *d_pos, *d_out;
    check_cuda(cudaMalloc(&d_img, h_image.size() * sizeof(float)), "alloc");
    check_cuda(cudaMalloc(&d_w, h_weight.size() * sizeof(float)), "alloc");
    check_cuda(cudaMalloc(&d_b, h_bias.size() * sizeof(float)), "alloc");
    check_cuda(cudaMalloc(&d_cls, h_cls.size() * sizeof(float)), "alloc");
    check_cuda(cudaMalloc(&d_pos, h_pos.size() * sizeof(float)), "alloc");
    check_cuda(cudaMalloc(&d_out, h_out_cpu.size() * sizeof(float)), "alloc");

    check_cuda(cudaMemcpy(d_img, h_image.data(), h_image.size() * sizeof(float), cudaMemcpyHostToDevice), "copy");
    check_cuda(cudaMemcpy(d_w, h_weight.data(), h_weight.size() * sizeof(float), cudaMemcpyHostToDevice), "copy");
    check_cuda(cudaMemcpy(d_b, h_bias.data(), h_bias.size() * sizeof(float), cudaMemcpyHostToDevice), "copy");
    check_cuda(cudaMemcpy(d_cls, h_cls.data(), h_cls.size() * sizeof(float), cudaMemcpyHostToDevice), "copy");
    check_cuda(cudaMemcpy(d_pos, h_pos.data(), h_pos.size() * sizeof(float), cudaMemcpyHostToDevice), "copy");

    patch_embedding_pipeline_cuda(d_img, d_w, d_b, d_cls, d_pos, d_out, shape);
    check_cuda(cudaDeviceSynchronize(), "sync");

    std::vector<float> h_out_gpu(h_out_cpu.size());
    check_cuda(cudaMemcpy(h_out_gpu.data(), d_out, h_out_gpu.size() * sizeof(float),
                          cudaMemcpyDeviceToHost), "copy");

    cudaFree(d_img); cudaFree(d_w); cudaFree(d_b);
    cudaFree(d_cls); cudaFree(d_pos); cudaFree(d_out);

    int mismatches = 0;
    for (std::size_t i = 0; i < h_out_cpu.size(); ++i) {
        if (!float_close(h_out_cpu[i], h_out_gpu[i], 1e-2f, 1e-3f)) {
            if (mismatches < 3) {
                std::printf("\n    [%zu] cpu=%.6f gpu=%.6f", i, h_out_cpu[i], h_out_gpu[i]);
            }
            ++mismatches;
        }
    }

    if (mismatches > 0) {
        std::printf("\n    FAIL (%d mismatches)\n", mismatches);
        return false;
    }
    std::printf("OK\n");
    return true;
}

// test: output shape
bool test_patch_embedding_shape() {
    std::printf("  test_patch_embedding_shape ... ");

    PatchEmbeddingShape shape;
    shape.batch       = 2;
    shape.in_channels = 1;
    shape.height      = 28;
    shape.width       = 28;
    shape.patch_size  = 7;
    shape.d_model     = 32;

    bool ok = true;
    ok = ok && (shape.n_patches() == 16);
    ok = ok && (shape.seq_len() == 17);
    ok = ok && (shape.patch_dim() == 49);
    ok = ok && (shape.output_elements() == 2 * 17 * 32);

    if (!ok) {
        std::printf("FAIL\n");
        return false;
    }
    std::printf("OK\n");
    return true;
}

int main() {
    std::printf("=== Tests: Patch Embedding ===\n");

    int passed = 0, total = 0;

    ++total; if (test_patch_embedding_shape()) ++passed;
    ++total; if (test_patch_embedding_cuda_vs_cpu()) ++passed;

    std::printf("\nResultado: %d/%d tests pasaron\n", passed, total);
    return passed == total ? 0 : 1;
}
