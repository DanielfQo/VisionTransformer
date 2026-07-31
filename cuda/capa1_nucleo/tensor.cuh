#pragma once

#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

// ------------------------------------------------------------
//
// Soporta formas N-dimensionales (en la practica: 2D, 3D, 4D):
//   · 4D (B, C, H, W) — imagenes de entrada
//   · 3D (B, N, d)    — secuencias de tokens
//   · 2D (B, d)       — logits / vectores de salida
// no modificar sin avisar al equipo.
// ------------------------------------------------------------

class Tensor {
public:

    explicit Tensor(std::vector<std::size_t> shape, float fill = 0.0f);

    Tensor(std::initializer_list<std::size_t> shape, float fill = 0.0f);

    std::size_t ndim() const;

    std::size_t dim(std::size_t i) const;

    //total de elementos
    std::size_t numel() const;

    //vector con todas las dimensiones
    const std::vector<std::size_t>& shape() const;

    // Acceso por indices multidimensionales (lectura).
    float at(std::initializer_list<std::size_t> idx) const;
    float at(const std::vector<std::size_t>& idx) const;

    float* data();
    const float* data() const;

    Tensor add(const Tensor& other) const;
    Tensor operator+(const Tensor& other) const;

    Tensor matmul(const Tensor& other) const;

    Tensor transpose() const;

    Tensor softmax() const;

    Tensor gelu() const;

    Tensor relu() const;

    Tensor scale(float s) const;
    Tensor operator*(float s) const;

    std::string shape_str() const;

private:
    std::vector<std::size_t> shape_;    // dimensiones
    std::vector<std::size_t> strides_;  // strides en row-major
    std::vector<float>       data_;     // datos contiguos

    // Calcula el indice plano
    std::size_t flat_index(const std::size_t* idx, std::size_t ndim) const;

    // Inicializa strides_
    void compute_strides();
};
