cmake_minimum_required(VERSION 3.18)
project(vit_cuda LANGUAGES CXX CUDA)

# ---- estandar ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# arquitectura CUDA (default: native si cmake >= 3.24, sino 75)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
        set(CMAKE_CUDA_ARCHITECTURES native)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 75)
    endif()
endif()

# ---- ruta a los datos MNIST ----
set(DATA_DIR "${CMAKE_SOURCE_DIR}/../data_mnist" CACHE PATH "Directorio con datos MNIST")

# ---- fuentes ----

# capa 1: nucleo (solo .cu que compila a .o)
set(CAPA1_SOURCES
    capa1_nucleo/tensor.cu
)

# capa 2: componentes
set(CAPA2_SOURCES
    capa2_componentes/patch_embedding.cu
    capa2_componentes/attention.cu
    capa2_componentes/encoder.cu
    capa2_componentes/softmax.cu
)

# capa 3: modelo
set(CAPA3_SOURCES
    capa3_modelo/vit.cu
)

# capa 4: entrenamiento (sin main)
set(CAPA4_SOURCES
    capa4_entrenamiento/mnist_loader.cu
    capa4_entrenamiento/training.cu
)

# ---- biblioteca estatica con todo el core ----
add_library(vit_core STATIC
    ${CAPA1_SOURCES}
    ${CAPA2_SOURCES}
    ${CAPA3_SOURCES}
    ${CAPA4_SOURCES}
)
target_include_directories(vit_core PUBLIC ${CMAKE_SOURCE_DIR})

# ---- ejecutable principal: entrenar ViT en MNIST ----
add_executable(vit_train
    capa4_entrenamiento/main.cu
)
target_link_libraries(vit_train PRIVATE vit_core)
target_compile_definitions(vit_train PRIVATE
    DATA_DIR="${DATA_DIR}"
)

# ---- tests ----
add_executable(test_attention tests/test_attention.cu)
target_link_libraries(test_attention PRIVATE vit_core)

add_executable(test_encoder tests/test_encoder.cu)
target_link_libraries(test_encoder PRIVATE vit_core)

add_executable(test_patch_embedding tests/test_patch_embedding.cu)
target_link_libraries(test_patch_embedding PRIVATE vit_core)

# target que ejecuta todos los tests
add_custom_target(run_tests
    COMMAND test_attention
    COMMAND test_encoder
    COMMAND test_patch_embedding
    DEPENDS test_attention test_encoder test_patch_embedding
    COMMENT "Ejecutando todos los tests..."
)
