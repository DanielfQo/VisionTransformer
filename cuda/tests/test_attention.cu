#include "../capa2_componentes/attention.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace vit;

bool float_close(float a, float b, float rtol = 1e-3f, float atol = 1e-5f) {
    return std::fabs(a - b) <= atol + rtol * std::fabs(b);
}

// test: la suma de pesos de atencion sobre K debe ser ~1.0 para cada query/head
bool test_attention_weights_sum() {
    std::printf("  test_attention_weights_sum ... ");

    const int batch = 1, tokens = 4, d_model = 8, heads = 2;
    AttentionShape shape{batch, tokens, d_model, heads};

    std::size_t n_elem = shape.token_elements();
    std::vector<float> h_input(n_elem);
    for (std::size_t i = 0; i < n_elem; ++i) {
        h_input[i] = static_cast<float>(i % 7) * 0.1f - 0.3f;
    }

    // CPU reference
    std::vector<float> weights;
    multi_head_self_attention_cpu(h_input, shape, nullptr, &weights);

    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            for (int q = 0; q < tokens; ++q) {
                float sum = 0.0f;
                for (int k = 0; k < tokens; ++k) {
                    std::size_t idx = ((static_cast<std::size_t>(b) * heads + h) * tokens + q) * tokens + k;
                    sum += weights[idx];
                }
                if (!float_close(sum, 1.0f, 1e-4f)) {
                    std::printf("FAIL (sum=%.6f)\n", sum);
                    return false;
                }
            }
        }
    }

    std::printf("OK\n");
    return true;
}

// test: CUDA vs CPU produce resultados cercanos
bool test_attention_cuda_vs_cpu() {
    std::printf("  test_attention_cuda_vs_cpu ... ");

    const int batch = 2, tokens = 5, d_model = 12, heads = 3;
    AttentionShape shape{batch, tokens, d_model, heads};

    std::size_t n_elem = shape.token_elements();
    std::vector<float> h_input(n_elem);
    for (std::size_t i = 0; i < n_elem; ++i) {
        h_input[i] = std::sin(static_cast<float>(i) * 0.1f);
    }

    // CPU
    std::vector<float> h_output_cpu =
        multi_head_self_attention_cpu(h_input, shape);

    // GPU
    float* d_input = nullptr;
    float* d_output = nullptr;
    check_cuda(cudaMalloc(&d_input, n_elem * sizeof(float)), "alloc input");
    check_cuda(cudaMalloc(&d_output, n_elem * sizeof(float)), "alloc output");
    check_cuda(cudaMemcpy(d_input, h_input.data(), n_elem * sizeof(float),
                          cudaMemcpyHostToDevice), "copy input");

    multi_head_self_attention_cuda(d_input, d_output, shape);
    check_cuda(cudaDeviceSynchronize(), "sync");

    std::vector<float> h_output_gpu(n_elem);
    check_cuda(cudaMemcpy(h_output_gpu.data(), d_output, n_elem * sizeof(float),
                          cudaMemcpyDeviceToHost), "copy output");

    cudaFree(d_input);
    cudaFree(d_output);

    int mismatches = 0;
    for (std::size_t i = 0; i < n_elem; ++i) {
        if (!float_close(h_output_cpu[i], h_output_gpu[i], 1e-2f, 1e-4f)) {
            if (mismatches < 3) {
                std::printf("\n    mismatch[%zu]: cpu=%.6f gpu=%.6f",
                            i, h_output_cpu[i], h_output_gpu[i]);
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

int main() {
    std::printf("=== Tests: Attention ===\n");

    int passed = 0, total = 0;

    ++total; if (test_attention_weights_sum()) ++passed;
    ++total; if (test_attention_cuda_vs_cpu()) ++passed;

    std::printf("\nResultado: %d/%d tests pasaron\n", passed, total);
    return passed == total ? 0 : 1;
}
