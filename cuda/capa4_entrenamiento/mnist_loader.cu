#include "mnist_loader.cuh"

#include <fstream>
#include <stdexcept>

namespace vit {

namespace {

// MNIST usa big-endian para los enteros de 32 bits
std::uint32_t read_u32_be(std::ifstream& file) {
    unsigned char bytes[4];
    file.read(reinterpret_cast<char*>(bytes), 4);
    if (!file) {
        throw std::runtime_error("Error al leer archivo MNIST");
    }
    return (static_cast<std::uint32_t>(bytes[0]) << 24) |
           (static_cast<std::uint32_t>(bytes[1]) << 16) |
           (static_cast<std::uint32_t>(bytes[2]) << 8)  |
           (static_cast<std::uint32_t>(bytes[3]));
}

}  // namespace

MnistDataset load_mnist(const std::string& image_path,
                         const std::string& label_path) {
    MnistDataset dataset;
    dataset.channels = 1;

    // ---- leer imagenes ----
    {
        std::ifstream file(image_path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("No se pudo abrir: " + image_path);
        }

        std::uint32_t magic = read_u32_be(file);
        if (magic != 0x00000803) {
            throw std::runtime_error("Magic number invalido en imagenes: " +
                                     std::to_string(magic));
        }

        std::uint32_t n_images = read_u32_be(file);
        std::uint32_t rows     = read_u32_be(file);
        std::uint32_t cols     = read_u32_be(file);

        dataset.n_samples = static_cast<int>(n_images);
        dataset.height    = static_cast<int>(rows);
        dataset.width     = static_cast<int>(cols);

        std::size_t total_pixels = static_cast<std::size_t>(n_images) * rows * cols;
        std::vector<unsigned char> raw(total_pixels);
        file.read(reinterpret_cast<char*>(raw.data()),
                  static_cast<std::streamsize>(total_pixels));
        if (!file) {
            throw std::runtime_error("Error al leer pixeles de: " + image_path);
        }

        // normalizar a [0, 1]
        dataset.images.resize(total_pixels);
        for (std::size_t i = 0; i < total_pixels; ++i) {
            dataset.images[i] = static_cast<float>(raw[i]) / 255.0f;
        }
    }

    // ---- leer labels ----
    {
        std::ifstream file(label_path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("No se pudo abrir: " + label_path);
        }

        std::uint32_t magic = read_u32_be(file);
        if (magic != 0x00000801) {
            throw std::runtime_error("Magic number invalido en labels: " +
                                     std::to_string(magic));
        }

        std::uint32_t n_labels = read_u32_be(file);
        if (static_cast<int>(n_labels) != dataset.n_samples) {
            throw std::runtime_error("Numero de labels no coincide con imagenes");
        }

        std::vector<unsigned char> raw(n_labels);
        file.read(reinterpret_cast<char*>(raw.data()),
                  static_cast<std::streamsize>(n_labels));
        if (!file) {
            throw std::runtime_error("Error al leer labels de: " + label_path);
        }

        dataset.labels.resize(n_labels);
        for (std::size_t i = 0; i < n_labels; ++i) {
            dataset.labels[i] = static_cast<int>(raw[i]);
        }
    }

    return dataset;
}

}  // namespace vit
