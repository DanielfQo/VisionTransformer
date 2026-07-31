#include "../capa2_componentes/softmax.cuh"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

bool close(float actual, float expected,
           float relative_tolerance = 1e-4f,
           float absolute_tolerance = 1e-6f) {
    return std::fabs(actual - expected) <=
           absolute_tolerance + relative_tolerance * std::fabs(expected);
}

std::vector<float> softmax_cpu(const std::vector<float>& input,
                               int rows,
                               int cols) {
    std::vector<float> output(input.size());
    for (int row = 0; row < rows; ++row) {
        const std::size_t offset = static_cast<std::size_t>(row) * cols;
        float maximum = input[offset];
        for (int col = 1; col < cols; ++col) {
            maximum = std::fmax(maximum, input[offset + col]);
        }

        float denominator = 0.0f;
        for (int col = 0; col < cols; ++col) {
            output[offset + col] = std::exp(input[offset + col] - maximum);
            denominator += output[offset + col];
        }
        for (int col = 0; col < cols; ++col) {
            output[offset + col] /= denominator;
        }
    }
    return output;
}

bool test_softmax_forward_and_stability() {
    std::printf("  test_softmax_forward_and_stability ... ");

    const int rows = 3;
    const int cols = 7;
    const std::vector<float> input{
        1000.0f, 1001.0f, 999.0f, 1002.0f, 998.0f, 997.0f, 996.0f,
        -1000.0f, -999.5f, -1001.0f, -998.0f, -1002.0f, -997.0f, -1003.0f,
        0.25f, -0.5f, 1.5f, 2.0f, -3.0f, 0.0f, 0.75f};
    const std::vector<float> expected = softmax_cpu(input, rows, cols);

    float* d_input = nullptr;
    float* d_output = nullptr;
    const std::size_t bytes = input.size() * sizeof(float);
    vit::check_cuda(cudaMalloc(&d_input, bytes), "alloc softmax input");
    vit::check_cuda(cudaMalloc(&d_output, bytes), "alloc softmax output");
    vit::check_cuda(cudaMemcpy(d_input, input.data(), bytes,
                               cudaMemcpyHostToDevice), "copy softmax input");

    vit::softmax_cuda(d_input, d_output, rows, cols);
    vit::check_cuda(cudaDeviceSynchronize(), "sync softmax forward");

    std::vector<float> actual(input.size());
    vit::check_cuda(cudaMemcpy(actual.data(), d_output, bytes,
                               cudaMemcpyDeviceToHost), "copy softmax output");
    cudaFree(d_input);
    cudaFree(d_output);

    for (int row = 0; row < rows; ++row) {
        float sum = 0.0f;
        for (int col = 0; col < cols; ++col) {
            const std::size_t index = static_cast<std::size_t>(row) * cols + col;
            if (!std::isfinite(actual[index]) ||
                !close(actual[index], expected[index])) {
                std::printf("FAIL (row=%d, col=%d, gpu=%.7f, cpu=%.7f)\n",
                            row, col, actual[index], expected[index]);
                return false;
            }
            sum += actual[index];
        }
        if (!close(sum, 1.0f)) {
            std::printf("FAIL (row=%d, sum=%.7f)\n", row, sum);
            return false;
        }
    }

    std::printf("OK\n");
    return true;
}

bool test_softmax_backward() {
    std::printf("  test_softmax_backward ... ");

    const int rows = 2;
    const int cols = 5;
    const std::vector<float> logits{
        1.0f, -2.0f, 0.5f, 3.0f, -0.25f,
        -1.0f, 0.0f, 2.0f, 1.5f, -3.0f};
    const std::vector<float> grad_out{
        0.2f, -0.4f, 1.0f, 0.3f, -0.1f,
        -0.5f, 0.7f, 0.1f, -0.2f, 0.9f};
    const std::vector<float> probabilities = softmax_cpu(logits, rows, cols);
    std::vector<float> expected(logits.size());

    for (int row = 0; row < rows; ++row) {
        const std::size_t offset = static_cast<std::size_t>(row) * cols;
        float dot = 0.0f;
        for (int col = 0; col < cols; ++col) {
            dot += probabilities[offset + col] * grad_out[offset + col];
        }
        for (int col = 0; col < cols; ++col) {
            expected[offset + col] =
                probabilities[offset + col] * (grad_out[offset + col] - dot);
        }
    }

    float* d_probabilities = nullptr;
    float* d_grad_out = nullptr;
    float* d_grad_in = nullptr;
    const std::size_t bytes = logits.size() * sizeof(float);
    vit::check_cuda(cudaMalloc(&d_probabilities, bytes), "alloc probabilities");
    vit::check_cuda(cudaMalloc(&d_grad_out, bytes), "alloc grad output");
    vit::check_cuda(cudaMalloc(&d_grad_in, bytes), "alloc grad input");
    vit::check_cuda(cudaMemcpy(d_probabilities, probabilities.data(), bytes,
                               cudaMemcpyHostToDevice), "copy probabilities");
    vit::check_cuda(cudaMemcpy(d_grad_out, grad_out.data(), bytes,
                               cudaMemcpyHostToDevice), "copy grad output");

    vit::softmax_backward_cuda(
        d_probabilities, d_grad_out, d_grad_in, rows, cols);
    vit::check_cuda(cudaDeviceSynchronize(), "sync softmax backward");

    std::vector<float> actual(logits.size());
    vit::check_cuda(cudaMemcpy(actual.data(), d_grad_in, bytes,
                               cudaMemcpyDeviceToHost), "copy grad input");
    cudaFree(d_probabilities);
    cudaFree(d_grad_out);
    cudaFree(d_grad_in);

    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (!close(actual[index], expected[index])) {
            std::printf("FAIL (index=%zu, gpu=%.7f, cpu=%.7f)\n",
                        index, actual[index], expected[index]);
            return false;
        }
    }

    std::printf("OK\n");
    return true;
}

}

int main() {
    std::printf("=== Tests: Softmax ===\n");

    int passed = 0;
    int total = 0;
    ++total;
    if (test_softmax_forward_and_stability()) {
        ++passed;
    }
    ++total;
    if (test_softmax_backward()) {
        ++passed;
    }

    std::printf("\nResultado: %d/%d tests pasaron\n", passed, total);
    return passed == total ? 0 : 1;
}
