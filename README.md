# Vision Transformer from scratch: Python, C++ and CUDA (MNIST)

A Vision Transformer (ViT) for image classification on **MNIST**, implemented three times without any deep-learning framework in the C++ and CUDA versions:

| Folder | What it is |
|---|---|
| [`python/`](python/) | Reference implementation. Used to define the architecture and to check the other two. |
| [`c++/`](c++/) | CPU implementation from scratch, with a layered design and a `Module` interface (PyTorch-style `forward`). Built by a team of four. |
| [`cuda/`](cuda/) | GPU implementation with hand-written kernels: fused multi-head attention, softmax, LayerNorm and GEMM, with forward and backward passes. |

## Highlights

- **Custom CUDA kernels** for softmax, LayerNorm and fused multi-head self-attention, using warp-shuffle reductions (`__shfl_down_sync`) and shared memory.
- **Forward and backward passes** implemented by hand, with GPU unit tests for each component (attention, encoder, patch embedding, softmax, gradients).
- **Measured optimizations** on an RTX 3050 Laptop GPU (see [Benchmarks](#benchmarks)).
- **Layered, testable design:** tensor core, components, model, training. Each layer depends only on the one below it.
- Default model: 28x28 input split into 7x7 patches, `d_model = 64`, 4 encoder layers, trained for 100 epochs (see `cuda/capa4_entrenamiento/main.cu`).

## Architecture

```
image (B, 1, 28, 28)
  -> Patch embedding   split into 7x7 patches, linear projection, [CLS] token, position embeddings
  -> Transformer encoder   N x ( LayerNorm -> Multi-head self-attention -> residual
                                 LayerNorm -> MLP (GELU)               -> residual )
  -> MLP head on the [CLS] token
  -> logits (B, 10)
```

Code layout (the same in `c++/` and `cuda/`):

- `capa1_nucleo/`: `Tensor` and the base `Module` interface. On GPU, also reduction helpers based on warp shuffle.
- `capa2_componentes/`: `patch_embedding`, `attention`, `encoder` (LayerNorm, MLP, encoder block) and, on GPU, `softmax`.
- `capa3_modelo/`: `vit`, the model that composes the components.
- `capa4_entrenamiento/`: cross-entropy loss, SGD update, MNIST IDX loader, training and evaluation loop, `main`.
- `tests/`: unit tests per component.

More detail on the C++ design: [`c++/docs/arquitectura.md`](c++/docs/arquitectura.md).

## Requirements

- NVIDIA GPU and CUDA Toolkit 11.x, 12.x or 13.x (for `cuda/`)
- CMake 3.20+
- MSVC (Windows) or GCC/G++ (Linux)

## Build and run (CUDA)

**Linux / WSL**

```bash
cd cuda
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build . --config Release
./vit_train
```

**Windows (PowerShell, Visual Studio Build Tools + Ninja)**

```powershell
cd cuda
cmd.exe /c "call ""C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"" && cmake -G ""Ninja"" -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86"
cmd.exe /c "call ""C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"" && cmake --build build --config Release"
.\build\vit_train.exe
```

Replace `86` with the compute capability of your GPU. To start from a clean build, run `cmake --build build --target clean` or delete the `build` folder.

## Tests

GPU unit tests, one per component:

```powershell
.\build\test_attention.exe
.\build\test_encoder.exe
.\build\test_patch_embedding.exe
```

The `cuda/tests/` folder also has `test_softmax.cu` and `test_gradients.cu`, which check the forward and backward passes.

## Benchmarks

Measured on an NVIDIA GeForce RTX 3050 Laptop GPU (compute capability 8.6, 16 SMs). Raw data and plots are in [`cuda/benchmarks/`](cuda/benchmarks/).

**Experiment 1: warp-shuffle reduction vs tree reduction** (selected sizes, time in microseconds)

| Kernel | Rows x cols | Tree | Shuffle | Speedup |
|---|---|---|---|---|
| LayerNorm | 256 x 256 | 57.67 | 22.15 | 2.60x |
| LayerNorm | 1024 x 256 | 100.96 | 40.37 | 2.50x |
| Softmax | 1024 x 256 | 90.82 | 42.52 | 2.14x |
| Softmax | 256 x 128 | 39.48 | 56.62 | 0.70x |

Shuffle reductions help most from 256 columns up; on the smallest sizes the gain is small or negative.

**Experiment 2: 16x16 tiled GEMM vs naive GEMM**

| M x K x N | Naive (us) | Tiled (us) | Speedup |
|---|---|---|---|
| 1024 x 512 x 512 | 3104.86 | 2288.27 | 1.36x |
| 1024 x 1024 x 1024 | 12411.40 | 9392.03 | 1.32x |
| 2048 x 1024 x 1024 | 26473.53 | 19876.45 | 1.33x |

Tiling gives roughly 1.3x on the larger matrices.

## Notes

- The code comments and some documents inside `c++/docs/` are in Spanish.
- This is an educational project: the goal is to understand and measure each piece of a Transformer on the GPU, not to compete with cuDNN or TensorRT.
