cmake_minimum_required(VERSION 3.20)
project(vit_cuda LANGUAGES CXX CUDA)

# ──────────────────────────────────────────────────────────────
# Estándares de lenguaje
# ──────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ──────────────────────────────────────────────────────────────
# Arquitectura GPU (ajusta según tu GPU):
#   Ampere     (RTX 30xx)   → 86
#   Ada        (RTX 40xx)   → 89
#   Hopper     (H100)       → 90
#   Blackwell  (RTX 50xx)   → 100
# ──────────────────────────────────────────────────────────────
set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING
    "Lista separada por punto y coma de versiones CUDA compute capability (ej. \"86;89\")")

# ──────────────────────────────────────────────────────────────
# Toolkit de CUDA
# ──────────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# ──────────────────────────────────────────────────────────────
# Ruta a los datos MNIST
# ──────────────────────────────────────────────────────────────
set(DATA_DIR "${CMAKE_SOURCE_DIR}/../data_mnist" CACHE PATH "Directorio con datos MNIST")

# ──────────────────────────────────────────────────────────────
# Recolección de fuentes
# ──────────────────────────────────────────────────────────────
set(CORE_SOURCES
    capa1_nucleo/tensor.cu
    capa2_componentes/patch_embedding.cu
    capa2_componentes/attention.cu
    capa2_componentes/encoder.cu
    capa2_componentes/softmax.cu
    capa3_modelo/vit.cu
    capa4_entrenamiento/mnist_loader.cu
    capa4_entrenamiento/training.cu
)

# ──────────────────────────────────────────────────────────────
# Librería Principal
# ──────────────────────────────────────────────────────────────
add_library(vit_core STATIC ${CORE_SOURCES})
target_include_directories(vit_core PUBLIC ${CMAKE_SOURCE_DIR})
target_compile_options(vit_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --extended-lambda
        --expt-relaxed-constexpr
        -Xcompiler=-Wall
    >
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
)

# ──────────────────────────────────────────────────────────────
# Ejecutable Principal (Entrenamiento / Inferencia ViT MNIST)
# ──────────────────────────────────────────────────────────────
add_executable(vit_train capa4_entrenamiento/main.cu)
target_link_libraries(vit_train PRIVATE vit_core)
target_compile_definitions(vit_train PRIVATE
    DATA_DIR="${DATA_DIR}"
)

# ──────────────────────────────────────────────────────────────
# Ejecutables de Tests
# ──────────────────────────────────────────────────────────────
add_executable(test_attention tests/test_attention.cu)
target_link_libraries(test_attention PRIVATE vit_core)

add_executable(test_encoder tests/test_encoder.cu)
target_link_libraries(test_encoder PRIVATE vit_core)

add_executable(test_patch_embedding tests/test_patch_embedding.cu)
target_link_libraries(test_patch_embedding PRIVATE vit_core)

# Target global para ejecutar todos los tests
add_custom_target(run_tests
    COMMAND test_attention
    COMMAND test_encoder
    COMMAND test_patch_embedding
    DEPENDS test_attention test_encoder test_patch_embedding
    COMMENT "Ejecutando todos los tests CUDA..."
)

# ──────────────────────────────────────────────────────────────
# Información del Build
# ──────────────────────────────────────────────────────────────
message(STATUS "CUDA Toolkit version : ${CUDAToolkit_VERSION}")
message(STATUS "CUDA arquitecturas   : ${CMAKE_CUDA_ARCHITECTURES}")
