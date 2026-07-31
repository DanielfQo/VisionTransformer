#include "../capa2_componentes/encoder.cuh"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace vit;

bool float_close(float a, float b, float rtol = 1e-3f, float atol = 1e-5f) {
    return std::fabs(a - b) <= atol + rtol * std::fabs(b);
}

// ---- test layer_norm ----
bool test_layer_norm_cuda_vs_cpu() {
    std::printf("  test_layer_norm_cuda_vs_cpu ... ");

    const int rows = 4, dim = 16;
    LayerNormShape shape{rows, dim};
    const float eps = 1e-5f;

    std::vector<float> h_input(shape.elements());
    std::vector<float> h_gamma(dim, 1.0f);
    std::vector<float> h_beta(dim, 0.0f);
    for (std::size_t i = 0; i < h_input.size(); ++i) {
        h_input[i] = std::sin(static_cast<float>(i) * 0.3f);
    }
    // non-trivial gamma/beta
    for (int i = 0; i < dim; ++i) {
        h_gamma[i] = 1.0f + 0.1f * i;
        h_beta[i] = -0.05f * i;
    }

    // CPU
    std::vector<float> h_output_cpu(shape.elements());
    layer_norm_cpu(h_input.data(), h_gamma.data(), h_beta.data(),
                   h_output_cpu.data(), shape, eps);

    // GPU
    float *d_in, *d_out, *d_gamma, *d_beta;
    std::size_t bytes = shape.elements() * sizeof(float);
    check_cuda(cudaMalloc(&d_in, bytes), "alloc");
    check_cuda(cudaMalloc(&d_out, bytes), "alloc");
    check_cuda(cudaMalloc(&d_gamma, dim * sizeof(float)), "alloc");
    check_cuda(cudaMalloc(&d_beta, dim * sizeof(float)), "alloc");

    check_cuda(cudaMemcpy(d_in, h_input.data(), bytes, cudaMemcpyHostToDevice), "copy");
    check_cuda(cudaMemcpy(d_gamma, h_gamma.data(), dim * sizeof(float), cudaMemcpyHostToDevice), "copy");
    check_cuda(cudaMemcpy(d_beta, h_beta.data(), dim * sizeof(float), cudaMemcpyHostToDevice), "copy");

    layer_norm_cuda(d_in, d_gamma, d_beta, d_out, shape, eps);
    check_cuda(cudaDeviceSynchronize(), "sync");

    std::vector<float> h_output_gpu(shape.elements());
    check_cuda(cudaMemcpy(h_output_gpu.data(), d_out, bytes, cudaMemcpyDeviceToHost), "copy");

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_gamma); cudaFree(d_beta);

    int mismatches = 0;
    for (std::size_t i = 0; i < h_output_cpu.size(); ++i) {
        if (!float_close(h_output_cpu[i], h_output_gpu[i], 1e-2f, 1e-4f)) {
            ++mismatches;
        }
    }

    if (mismatches > 0) {
        std::printf("FAIL (%d mismatches)\n", mismatches);
        return false;
    }
    std::printf("OK\n");
    return true;
}

// ---- test linear ----
bool test_linear_cuda_vs_cpu() {
    std::printf("  test_linear_cuda_vs_cpu ... ");

    const int rows = 3, in_f = 8, out_f = 4;
    LinearShape shape{rows, in_f, out_f};

    std::vector<float> h_input(shape.input_elements());
    std::vector<float> h_weight(static_cast<std::size_t>(in_f) * out_f);
    std::vector<float> h_bias(out_f, 0.5f);

    for (std::size_t i = 0; i < h_input.size(); ++i) h_input[i] = (float)i * 0.1f;
    for (std::size_t i = 0; i < h_weight.size(); ++i) h_weight[i] = std::cos((float)i * 0.2f);

    // CPU
    std::vector<float> h_out_cpu(shape.output_elements());
    linear_cpu(h_input.data(), h_weight.data(), h_bias.data(),
               h_out_cpu.data(), shape);

    // GPU
    float *d_in, *d_w, *d_b, *d_out;
    check_cuda(cudaMalloc(&d_in, h_input.size() * sizeof(float)), "alloc");
    check_cuda(cudaMalloc(&d_w, h_weight.size() * sizeof(float)), "alloc");
    check_cuda(cudaMalloc(&d_b, h_bias.size() * sizeof(float)), "alloc");
    check_cuda(cudaMalloc(&d_out, h_out_cpu.size() * sizeof(float)), "alloc");

    check_cuda(cudaMemcpy(d_in, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice), "copy");
    check_cuda(cudaMemcpy(d_w, h_weight.data(), h_weight.size() * sizeof(float), cudaMemcpyHostToDevice), "copy");
    check_cuda(cudaMemcpy(d_b, h_bias.data(), h_bias.size() * sizeof(float), cudaMemcpyHostToDevice), "copy");

    linear_cuda(d_in, d_w, d_b, d_out, shape);
    check_cuda(cudaDeviceSynchronize(), "sync");

    std::vector<float> h_out_gpu(h_out_cpu.size());
    check_cuda(cudaMemcpy(h_out_gpu.data(), d_out, h_out_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy");

    cudaFree(d_in); cudaFree(d_w); cudaFree(d_b); cudaFree(d_out);

    int mismatches = 0;
    for (std::size_t i = 0; i < h_out_cpu.size(); ++i) {
        if (!float_close(h_out_cpu[i], h_out_gpu[i], 1e-2f, 1e-3f)) {
            ++mismatches;
        }
    }

    if (mismatches > 0) {
        std::printf("FAIL (%d mismatches)\n", mismatches);
        return false;
    }
    std::printf("OK\n");
    return true;
}

// ---- test gelu ----
bool test_gelu_cuda_vs_cpu() {
    std::printf("  test_gelu_cuda_vs_cpu ... ");

    const std::size_t n = 64;
    std::vector<float> h_input(n);
    for (std::size_t i = 0; i < n; ++i) {
        h_input[i] = static_cast<float>(i) * 0.1f - 3.0f;
    }

    // CPU
    std::vector<float> h_out_cpu(n);
    gelu_cpu(h_input.data(), h_out_cpu.data(), n);

    // GPU
    float *d_in, *d_out;
    check_cuda(cudaMalloc(&d_in, n * sizeof(float)), "alloc");
    check_cuda(cudaMalloc(&d_out, n * sizeof(float)), "alloc");
    check_cuda(cudaMemcpy(d_in, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice), "copy");

    gelu_cuda(d_in, d_out, n);
    check_cuda(cudaDeviceSynchronize(), "sync");

    std::vector<float> h_out_gpu(n);
    check_cuda(cudaMemcpy(h_out_gpu.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost), "copy");

    cudaFree(d_in); cudaFree(d_out);

    int mismatches = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (!float_close(h_out_cpu[i], h_out_gpu[i], 1e-3f, 1e-5f)) {
            ++mismatches;
        }
    }

    if (mismatches > 0) {
        std::printf("FAIL (%d mismatches)\n", mismatches);
        return false;
    }
    std::printf("OK\n");
    return true;
}

int main() {
    std::printf("=== Tests: Encoder (LayerNorm, Linear, GELU) ===\n");

    int passed = 0, total = 0;

    ++total; if (test_layer_norm_cuda_vs_cpu()) ++passed;
    ++total; if (test_linear_cuda_vs_cpu()) ++passed;
    ++total; if (test_gelu_cuda_vs_cpu()) ++passed;

    std::printf("\nResultado: %d/%d tests pasaron\n", passed, total);
    return passed == total ? 0 : 1;
}
