// ============================================================================
// bench_optimizations.cu
//
// Experimento 1: Warp Shuffle vs. Tree Reduction (LayerNorm y Softmax)
// Experimento 2: Tiling 16x16 vs. GEMM Naive (Linear)
//
// Salida: CSV por stdout para graficar con Python.
// ============================================================================

#include "../capa1_nucleo/cuda_utils.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <string>

// ============================================================================
// Utilidad de benchmarking con cudaEvent_t
// ============================================================================

struct BenchResult {
    float time_ms;
};

template<typename Func>
BenchResult benchmark_kernel(Func fn, int warmup_runs = 10, int timed_runs = 100) {
    for (int i = 0; i < warmup_runs; ++i) fn();
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < timed_runs; ++i) fn();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return {total_ms / static_cast<float>(timed_runs)};
}

// ============================================================================
// EXPERIMENTO 1: Kernels alternativos con Tree Reduction (sin Warp Shuffle)
// ============================================================================

namespace tree {

// Reduccion clasica por arbol en shared memory (sin warp shuffle)
__device__ float block_reduce_sum_tree(float val) {
    __shared__ float shared[256];
    shared[threadIdx.x] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < static_cast<unsigned>(s)) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }
    return shared[0];
}

__device__ float block_reduce_max_tree(float val) {
    __shared__ float shared[256];
    shared[threadIdx.x] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < static_cast<unsigned>(s)) {
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + s]);
        }
        __syncthreads();
    }
    return shared[0];
}

// LayerNorm con tree reduction
__global__ void layer_norm_tree_kernel(const float* input, const float* gamma,
                                       const float* beta, float* output,
                                       int dim, float eps) {
    int row = blockIdx.x;
    const float* x = input + static_cast<std::size_t>(row) * dim;
    float* y = output + static_cast<std::size_t>(row) * dim;

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        local_sum += x[i];
    }
    float mean = block_reduce_sum_tree(local_sum) / static_cast<float>(dim);

    float local_var = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float d = x[i] - mean;
        local_var += d * d;
    }
    float inv = rsqrtf(block_reduce_sum_tree(local_var) / static_cast<float>(dim) + eps);

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        y[i] = (x[i] - mean) * inv * gamma[i] + beta[i];
    }
}

// Softmax con tree reduction
__global__ void softmax_tree_kernel(const float* input, float* output, int cols) {
    int row = blockIdx.x;
    const float* x = input + static_cast<std::size_t>(row) * cols;
    float* y = output + static_cast<std::size_t>(row) * cols;

    float local_max = -CUDART_INF_F;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        local_max = fmaxf(local_max, x[i]);
    }
    float max_val = block_reduce_max_tree(local_max);

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float e = expf(x[i] - max_val);
        y[i] = e;
        local_sum += e;
    }
    float total = block_reduce_sum_tree(local_sum);

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        y[i] /= total;
    }
}

}  // namespace tree

// ============================================================================
// EXPERIMENTO 1: Kernels optimizados con Warp Shuffle (los actuales del proyecto)
// ============================================================================

namespace shuffle {

// Reutilizamos block_reduce_sum y block_reduce_max de cuda_utils.cuh (namespace vit)

__global__ void layer_norm_shuffle_kernel(const float* input, const float* gamma,
                                           const float* beta, float* output,
                                           int dim, float eps) {
    using namespace vit;
    int row = blockIdx.x;
    const float* x = input + static_cast<std::size_t>(row) * dim;
    float* y = output + static_cast<std::size_t>(row) * dim;

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        local_sum += x[i];
    }
    float total_sum = block_reduce_sum(local_sum);
    __shared__ float s_mean;
    if (threadIdx.x == 0) {
        s_mean = total_sum / static_cast<float>(dim);
    }
    __syncthreads();
    float mean = s_mean;

    float local_var = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float d = x[i] - mean;
        local_var += d * d;
    }
    float total_var = block_reduce_sum(local_var);
    __shared__ float s_inv;
    if (threadIdx.x == 0) {
        s_inv = rsqrtf(total_var / static_cast<float>(dim) + eps);
    }
    __syncthreads();
    float inv = s_inv;

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        y[i] = (x[i] - mean) * inv * gamma[i] + beta[i];
    }
}

__global__ void softmax_shuffle_kernel(const float* input, float* output, int cols) {
    using namespace vit;
    int row = blockIdx.x;
    const float* x = input + static_cast<std::size_t>(row) * cols;
    float* y = output + static_cast<std::size_t>(row) * cols;

    float local_max = -CUDART_INF_F;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        local_max = fmaxf(local_max, x[i]);
    }
    float max_val = block_reduce_max(local_max);
    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = max_val;
    __syncthreads();
    max_val = s_max;

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float e = expf(x[i] - max_val);
        y[i] = e;
        local_sum += e;
    }
    float total = block_reduce_sum(local_sum);
    __shared__ float s_total;
    if (threadIdx.x == 0) s_total = total;
    __syncthreads();
    total = s_total;

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        y[i] /= total;
    }
}

}  // namespace shuffle

// ============================================================================
// EXPERIMENTO 2: GEMM Naive (sin shared memory tiling)
// ============================================================================

namespace naive_gemm {

__global__ void linear_naive_kernel(const float* input, const float* weight,
                                     const float* bias, float* output,
                                     int in_features, int out_features, int rows) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || col >= out_features) return;

    float acc = bias != nullptr ? bias[col] : 0.0f;
    for (int k = 0; k < in_features; ++k) {
        acc += input[static_cast<std::size_t>(row) * in_features + k]
             * weight[static_cast<std::size_t>(k) * out_features + col];
    }
    output[static_cast<std::size_t>(row) * out_features + col] = acc;
}

}  // namespace naive_gemm

// ============================================================================
// EXPERIMENTO 2: GEMM con Tiling 16x16 en Shared Memory (la version optimizada)
// ============================================================================

namespace tiled_gemm {

constexpr int kTileSize = 16;

__global__ void linear_tiled_kernel(const float* input, const float* weight,
                                     const float* bias, float* output,
                                     int in_features, int out_features, int rows) {
    __shared__ float s_in[kTileSize][kTileSize];
    __shared__ float s_w[kTileSize][kTileSize];

    int col = blockIdx.x * kTileSize + threadIdx.x;
    int row = blockIdx.y * kTileSize + threadIdx.y;

    float acc = 0.0f;
    int num_tiles = (in_features + kTileSize - 1) / kTileSize;

    for (int t = 0; t < num_tiles; ++t) {
        int in_col = t * kTileSize + threadIdx.x;
        if (row < rows && in_col < in_features) {
            s_in[threadIdx.y][threadIdx.x] = input[static_cast<std::size_t>(row) * in_features + in_col];
        } else {
            s_in[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int in_row = t * kTileSize + threadIdx.y;
        if (in_row < in_features && col < out_features) {
            s_w[threadIdx.y][threadIdx.x] = weight[static_cast<std::size_t>(in_row) * out_features + col];
        } else {
            s_w[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < kTileSize; ++k) {
            acc += s_in[threadIdx.y][k] * s_w[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < rows && col < out_features) {
        if (bias != nullptr) {
            acc += bias[col];
        }
        output[static_cast<std::size_t>(row) * out_features + col] = acc;
    }
}

}  // namespace tiled_gemm


// ============================================================================
// Funciones auxiliares
// ============================================================================

void fill_random(float* d_ptr, std::size_t n, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> host(n);
    for (auto& v : host) v = dist(rng);
    cudaMemcpy(d_ptr, host.data(), n * sizeof(float), cudaMemcpyHostToDevice);
}

void fill_ones(float* d_ptr, std::size_t n) {
    std::vector<float> host(n, 1.0f);
    cudaMemcpy(d_ptr, host.data(), n * sizeof(float), cudaMemcpyHostToDevice);
}

void fill_zeros(float* d_ptr, std::size_t n) {
    cudaMemset(d_ptr, 0, n * sizeof(float));
}

// ============================================================================
// main: Ejecuta ambos experimentos y escribe CSV
// ============================================================================

int main() {
    // Informacion de la GPU
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::printf("GPU: %s\n", prop.name);
    std::printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    std::printf("SMs: %d\n\n", prop.multiProcessorCount);

    constexpr int kBlock = 256;

    // ==================================================================
    // EXPERIMENTO 1: Warp Shuffle vs. Tree Reduction
    // ==================================================================
    std::printf("=== EXPERIMENTO 1: Warp Shuffle vs Tree Reduction ===\n");
    std::printf("CSV_EXP1_START\n");
    std::printf("kernel,rows,cols,tree_us,shuffle_us,speedup\n");

    struct Exp1Config { int rows; int cols; };
    Exp1Config exp1_configs[] = {
        {256,   64},
        {256,  128},
        {256,  256},
        {512,  256},
        {1024, 256},
        {1024, 512},
        {2048, 256},
        {2048, 512},
    };

    for (auto& cfg : exp1_configs) {
        int rows = cfg.rows;
        int cols = cfg.cols;
        std::size_t n = static_cast<std::size_t>(rows) * cols;

        float *d_input, *d_output_tree, *d_output_shuffle, *d_gamma, *d_beta;
        cudaMalloc(&d_input, n * sizeof(float));
        cudaMalloc(&d_output_tree, n * sizeof(float));
        cudaMalloc(&d_output_shuffle, n * sizeof(float));
        cudaMalloc(&d_gamma, cols * sizeof(float));
        cudaMalloc(&d_beta, cols * sizeof(float));

        fill_random(d_input, n);
        fill_ones(d_gamma, cols);
        fill_zeros(d_beta, cols);

        // --- LayerNorm ---
        auto ln_tree = benchmark_kernel([&]() {
            tree::layer_norm_tree_kernel<<<rows, kBlock>>>(
                d_input, d_gamma, d_beta, d_output_tree, cols, 1e-5f);
        });
        auto ln_shuffle = benchmark_kernel([&]() {
            shuffle::layer_norm_shuffle_kernel<<<rows, kBlock>>>(
                d_input, d_gamma, d_beta, d_output_shuffle, cols, 1e-5f);
        });
        float ln_speedup = ln_tree.time_ms / ln_shuffle.time_ms;
        std::printf("LayerNorm,%d,%d,%.2f,%.2f,%.2f\n",
                    rows, cols,
                    ln_tree.time_ms * 1000.0f,
                    ln_shuffle.time_ms * 1000.0f,
                    ln_speedup);

        // --- Softmax ---
        auto sm_tree = benchmark_kernel([&]() {
            tree::softmax_tree_kernel<<<rows, kBlock>>>(
                d_input, d_output_tree, cols);
        });
        auto sm_shuffle = benchmark_kernel([&]() {
            shuffle::softmax_shuffle_kernel<<<rows, kBlock>>>(
                d_input, d_output_shuffle, cols);
        });
        float sm_speedup = sm_tree.time_ms / sm_shuffle.time_ms;
        std::printf("Softmax,%d,%d,%.2f,%.2f,%.2f\n",
                    rows, cols,
                    sm_tree.time_ms * 1000.0f,
                    sm_shuffle.time_ms * 1000.0f,
                    sm_speedup);

        cudaFree(d_input);
        cudaFree(d_output_tree);
        cudaFree(d_output_shuffle);
        cudaFree(d_gamma);
        cudaFree(d_beta);
    }
    std::printf("CSV_EXP1_END\n\n");

    // ==================================================================
    // EXPERIMENTO 2: Tiling vs. GEMM Naive
    // ==================================================================
    std::printf("=== EXPERIMENTO 2: Tiling 16x16 vs GEMM Naive ===\n");
    std::printf("CSV_EXP2_START\n");
    std::printf("M,K,N,naive_us,tiled_us,speedup\n");

    struct Exp2Config { int M; int K; int N; };
    Exp2Config exp2_configs[] = {
        // Dimensiones tipicas del ViT (batch*seq_len, d_model, mlp_hidden)
        {320,   64,  256},   // batch=64 * seq=5, d_model=64, mlp=256
        {320,  256,   64},   // la inversa: mlp -> d_model
        {1088,  64,  256},   // batch=64 * seq=17, d_model=64, mlp=256
        {1088, 256,   64},   // la inversa
        // Dimensiones mas grandes para ver efecto del tiling
        {512,  256,  256},
        {512,  512,  512},
        {1024, 256,  256},
        {1024, 512,  512},
        {1024,1024, 1024},
        {2048,1024, 1024},
    };

    for (auto& cfg : exp2_configs) {
        int M = cfg.M, K = cfg.K, N = cfg.N;
        std::size_t size_A = static_cast<std::size_t>(M) * K;
        std::size_t size_B = static_cast<std::size_t>(K) * N;
        std::size_t size_C = static_cast<std::size_t>(M) * N;

        float *d_A, *d_B, *d_bias, *d_C_naive, *d_C_tiled;
        cudaMalloc(&d_A, size_A * sizeof(float));
        cudaMalloc(&d_B, size_B * sizeof(float));
        cudaMalloc(&d_bias, N * sizeof(float));
        cudaMalloc(&d_C_naive, size_C * sizeof(float));
        cudaMalloc(&d_C_tiled, size_C * sizeof(float));

        fill_random(d_A, size_A, 1);
        fill_random(d_B, size_B, 2);
        fill_random(d_bias, N, 3);

        // Naive GEMM
        constexpr int kTile = 16;
        dim3 block_naive(kTile, kTile);
        dim3 grid_naive((N + kTile - 1) / kTile, (M + kTile - 1) / kTile);

        auto res_naive = benchmark_kernel([&]() {
            naive_gemm::linear_naive_kernel<<<grid_naive, block_naive>>>(
                d_A, d_B, d_bias, d_C_naive, K, N, M);
        });

        // Tiled GEMM
        dim3 block_tiled(kTile, kTile);
        dim3 grid_tiled((N + kTile - 1) / kTile, (M + kTile - 1) / kTile);

        auto res_tiled = benchmark_kernel([&]() {
            tiled_gemm::linear_tiled_kernel<<<grid_tiled, block_tiled>>>(
                d_A, d_B, d_bias, d_C_tiled, K, N, M);
        });

        float speedup = res_naive.time_ms / res_tiled.time_ms;
        std::printf("%d,%d,%d,%.2f,%.2f,%.2f\n",
                    M, K, N,
                    res_naive.time_ms * 1000.0f,
                    res_tiled.time_ms * 1000.0f,
                    speedup);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_bias);
        cudaFree(d_C_naive);
        cudaFree(d_C_tiled);
    }
    std::printf("CSV_EXP2_END\n\n");

    std::printf("Benchmarks completados.\n");
    return 0;
}
