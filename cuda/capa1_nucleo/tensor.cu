#include "tensor.cuh"

#include <algorithm>   // std::max
#include <cmath>       // std::exp, std::tanh, std::sqrt
#include <sstream>     // std::ostringstream
#include <stdexcept>
#include <string>


namespace {

// Valida que dos formas sean iguales
void require_same_shape(const std::vector<std::size_t>& a,
                        const std::vector<std::size_t>& b,
                        const char* op) {
    if (a != b) {
        std::ostringstream msg;
        msg << op << ": las formas no coinciden.";
        throw std::invalid_argument(msg.str());
    }
}

}



Tensor::Tensor(std::vector<std::size_t> shape, float fill)
    : shape_(std::move(shape)) {
    compute_strides();
    std::size_t n = numel();
    data_.assign(n, fill);
}

Tensor::Tensor(std::initializer_list<std::size_t> shape, float fill)
    : shape_(shape) {
    compute_strides();
    std::size_t n = numel();
    data_.assign(n, fill);
}



std::size_t Tensor::ndim() const {
    return shape_.size();
}

std::size_t Tensor::dim(std::size_t i) const {
    if (i >= shape_.size()) {
        throw std::out_of_range("Tensor::dim — indice fuera de rango");
    }
    return shape_[i];
}

std::size_t Tensor::numel() const {
    if (shape_.empty()) return 0;
    std::size_t n = 1;
    for (std::size_t d : shape_) {
        n *= d;
    }
    return n;
}

const std::vector<std::size_t>& Tensor::shape() const {
    return shape_;
}

std::size_t Tensor::flat_index(const std::size_t* idx,
                                std::size_t ndim) const {
    if (ndim != shape_.size()) {
        throw std::invalid_argument(
            "Tensor::at — numero de indices no coincide con el rango del tensor");
    }
    std::size_t flat = 0;
    for (std::size_t i = 0; i < ndim; ++i) {
        if (idx[i] >= shape_[i]) {
            throw std::out_of_range(
                "Tensor::at — indice fuera de rango en dimension " +
                std::to_string(i));
        }
        flat += idx[i] * strides_[i];
    }
    return flat;
}

float Tensor::at(std::initializer_list<std::size_t> idx) const {
    return data_[flat_index(idx.begin(), idx.size())];
}

float Tensor::at(const std::vector<std::size_t>& idx) const {
    return data_[flat_index(idx.data(), idx.size())];
}

float* Tensor::data() {
    return data_.data();
}

const float* Tensor::data() const {
    return data_.data();
}

// ============================================================
// Strides
// ============================================================

void Tensor::compute_strides() {
    const std::size_t n = shape_.size();
    strides_.resize(n);
    if (n == 0) return;
    strides_[n - 1] = 1;
    for (std::size_t i = n - 1; i > 0; --i) {
        strides_[i - 1] = strides_[i] * shape_[i];
    }
}



Tensor Tensor::add(const Tensor& other) const {
    require_same_shape(shape_, other.shape_, "Tensor::add");
    Tensor out(shape_);
    const std::size_t n = numel();
    for (std::size_t i = 0; i < n; ++i) {
        out.data_[i] = data_[i] + other.data_[i];
    }
    return out;
}

Tensor Tensor::operator+(const Tensor& other) const {
    return add(other);
}

// ---- Matmul ----
// Soporta: 2D (M,K)·(K,N)→(M,N)
//          3D (B,M,K)·(B,K,N)→(B,M,N)
//          4D (B,H,M,K)·(B,H,K,N)→(B,H,M,N)
// Las dimensiones del batch deben coincidir; las ultimas 2 son la matriz.

Tensor Tensor::matmul(const Tensor& other) const {
    const std::size_t ra = ndim();
    const std::size_t rb = other.ndim();

    if (ra < 2 || rb < 2) {
        throw std::invalid_argument("Tensor::matmul — ambos tensores deben ser al menos 2D");
    }
    if (ra != rb) {
        throw std::invalid_argument("Tensor::matmul — los tensores deben tener el mismo rango");
    }


    for (std::size_t i = 0; i < ra - 2; ++i) {
        if (shape_[i] != other.shape_[i]) {
            throw std::invalid_argument(
                "Tensor::matmul — las dimensiones de batch no coinciden");
        }
    }

    const std::size_t M = shape_[ra - 2];
    const std::size_t K = shape_[ra - 1];
    const std::size_t K2 = other.shape_[rb - 2];
    const std::size_t N = other.shape_[rb - 1];

    if (K != K2) {
        throw std::invalid_argument(
            "Tensor::matmul — dimensiones internas incompatibles: " +
            std::to_string(K) + " vs " + std::to_string(K2));
    }

    // Construir la forma del resultado
    std::vector<std::size_t> out_shape(shape_.begin(), shape_.begin() + static_cast<long>(ra - 2));
    out_shape.push_back(M);
    out_shape.push_back(N);
    Tensor out(out_shape, 0.0f);

    // Numero de "matrices" batch = producto de todas las dims de batch
    std::size_t batch = 1;
    for (std::size_t i = 0; i < ra - 2; ++i) batch *= shape_[i];

    // Stride de matrix en cada tensor
    const std::size_t stride_a = M * K;
    const std::size_t stride_b = K * N;
    const std::size_t stride_c = M * N;

    for (std::size_t b = 0; b < batch; ++b) {
        const float* A = data_.data() + b * stride_a;
        const float* B = other.data_.data() + b * stride_b;
        float* C = out.data_.data() + b * stride_c;

        for (std::size_t m = 0; m < M; ++m) {
            for (std::size_t n = 0; n < N; ++n) {
                float acc = 0.0f;
                for (std::size_t k = 0; k < K; ++k) {
                    acc += A[m * K + k] * B[k * N + n];
                }
                C[m * N + n] = acc;
            }
        }
    }

    return out;
}

// ---- Transpuesta (ultimas 2 dimensiones) ----

Tensor Tensor::transpose() const {
    if (ndim() < 2) {
        throw std::invalid_argument("Tensor::transpose — se necesita al menos rango 2");
    }

    const std::size_t r = ndim();
    const std::size_t M = shape_[r - 2];
    const std::size_t N = shape_[r - 1];

    std::vector<std::size_t> out_shape(shape_);
    out_shape[r - 2] = N;
    out_shape[r - 1] = M;
    Tensor out(out_shape, 0.0f);

    std::size_t batch = 1;
    for (std::size_t i = 0; i < r - 2; ++i) batch *= shape_[i];

    const std::size_t stride_mat = M * N;

    for (std::size_t b = 0; b < batch; ++b) {
        const float* src = data_.data() + b * stride_mat;
        float* dst = out.data_.data() + b * stride_mat;

        for (std::size_t m = 0; m < M; ++m) {
            for (std::size_t n = 0; n < N; ++n) {
                dst[n * M + m] = src[m * N + n];
            }
        }
    }

    return out;
}

// ---- Softmax (sobre el ultimo eje) ----
// Numericamente estable: softmax(x - max(x))

Tensor Tensor::softmax() const {
    if (ndim() == 0) {
        throw std::invalid_argument("Tensor::softmax — tensor vacio");
    }

    const std::size_t last = shape_.back();         // tamano del eje softmax
    const std::size_t outer = numel() / last;       // producto de los demas ejes

    Tensor out(shape_, 0.0f);

    for (std::size_t i = 0; i < outer; ++i) {
        const float* src = data_.data() + i * last;
        float* dst = out.data_.data() + i * last;

        // Maximo para estabilidad numerica
        float max_val = src[0];
        for (std::size_t j = 1; j < last; ++j) {
            if (src[j] > max_val) max_val = src[j];
        }

        float sum = 0.0f;
        for (std::size_t j = 0; j < last; ++j) {
            dst[j] = std::exp(src[j] - max_val);
            sum += dst[j];
        }
        for (std::size_t j = 0; j < last; ++j) {
            dst[j] /= sum;
        }
    }

    return out;
}

// ---- GELU (aproximacion) ----

Tensor Tensor::gelu() const {
    // Constante √(2/π)
    static constexpr float kSqrt2OverPi = 0.7978845608028654f;
    static constexpr float kCoeff       = 0.044715f;

    Tensor out(shape_, 0.0f);
    const std::size_t n = numel();

    for (std::size_t i = 0; i < n; ++i) {
        const float x = data_[i];
        const float inner = kSqrt2OverPi * (x + kCoeff * x * x * x);
        out.data_[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }

    return out;
}



Tensor Tensor::relu() const {
    Tensor out(shape_, 0.0f);
    const std::size_t n = numel();
    for (std::size_t i = 0; i < n; ++i) {
        out.data_[i] = data_[i] > 0.0f ? data_[i] : 0.0f;
    }
    return out;
}


Tensor Tensor::scale(float s) const {
    Tensor out(shape_, 0.0f);
    const std::size_t n = numel();
    for (std::size_t i = 0; i < n; ++i) {
        out.data_[i] = data_[i] * s;
    }
    return out;
}

Tensor Tensor::operator*(float s) const {
    return scale(s);
}

std::string Tensor::shape_str() const {
    std::ostringstream oss;
    oss << "(";
    for (std::size_t i = 0; i < shape_.size(); ++i) {
        oss << shape_[i];
        if (i + 1 < shape_.size()) oss << ", ";
    }
    oss << ")";
    return oss.str();
}
