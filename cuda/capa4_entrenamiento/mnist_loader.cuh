#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vit {

struct MnistDataset {
    std::vector<float> images;       // [n_samples, 1, 28, 28] normalizadas a [0,1]
    std::vector<int>   labels;       // [n_samples]
    int n_samples;
    int height;
    int width;
    int channels;
};

// carga imagenes y labels desde archivos IDX de MNIST
MnistDataset load_mnist(const std::string& image_path,
                         const std::string& label_path);

}  // namespace vit
