#include "mnist_loader.cuh"
#include "training.cuh"
#include "../capa3_modelo/vit.cuh"

#include <cstdio>
#include <string>

#ifndef DATA_DIR
#define DATA_DIR "../data_mnist"
#endif

int main() {

    // ---- 1. Cargar datos ----
    std::string data_dir = DATA_DIR;

    vit::MnistDataset train_data;
    vit::MnistDataset test_data;

    try {
        train_data = vit::load_mnist(data_dir + "/train-images-idx3-ubyte",
                                    data_dir + "/train-labels-idx1-ubyte");
        test_data  = vit::load_mnist(data_dir + "/t10k-images-idx3-ubyte",
                                    data_dir + "/t10k-labels-idx1-ubyte");
        std::printf("Cargando datos desde: %s\n", data_dir.c_str());
        std::printf("  Dataset original train: %d imagenes (%dx%d)\n",
                    train_data.n_samples, train_data.width, train_data.height);
        std::printf("  Dataset test:           %d imagenes (%dx%d)\n\n",
                    test_data.n_samples, test_data.width, test_data.height);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error cargando MNIST: %s\n", e.what());
        return 1;
    }

    // ---- 2. Configurar modelo ----
    vit::VitConfig config;
    config.image_size  = 28;
    config.patch_size  = 7;
    config.in_channels = 1;
    config.d_model     = 64;
    config.heads       = 4;
    config.n_layers    = 4;
    config.mlp_hidden  = 256;
    config.n_classes   = 10;

    // particionar train_data (60,000) en 50,000 train y 10,000 val
    int n_train = 50000;
    int n_val   = 10000;
    int img_size = config.in_channels * config.image_size * config.image_size;

    const float* train_img_ptr = train_data.images.data();
    const int*   train_lbl_ptr = train_data.labels.data();

    const float* val_img_ptr   = train_data.images.data() + static_cast<std::size_t>(n_train) * img_size;
    const int*   val_lbl_ptr   = train_data.labels.data() + n_train;

    std::printf("Configuracion del ViT:\n");
    std::printf("  Imagen:      %dx%dx%d\n", config.image_size, config.image_size, config.in_channels);
    std::printf("  Patch size:  %d\n", config.patch_size);
    std::printf("  Patches:     %d\n", config.n_patches());
    std::printf("  Seq length:  %d (patches + CLS)\n", config.seq_len());
    std::printf("  d_model:     %d\n", config.d_model);
    std::printf("  Heads:       %d\n", config.heads);
    std::printf("  Layers:      %d\n", config.n_layers);
    std::printf("  MLP hidden:  %d\n", config.mlp_hidden);
    std::printf("  Classes:     %d\n\n", config.n_classes);
    std::printf("Particion de datos:\n");
    std::printf("  Entrenamiento: %d imagenes\n", n_train);
    std::printf("  Validacion:    %d imagenes\n", n_val);
    std::printf("  Prueba (test): %d imagenes\n\n", test_data.n_samples);

    // ---- 3. Inicializar pesos ----
    vit::VitWeights weights;
    try {
        vit::vit_init_weights(weights, config, 42);
        std::printf("Pesos inicializados (Xavier uniform, seed=42)\n\n");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error inicializando pesos: %s\n", e.what());
        return 1;
    }

    // ---- 4. Entrenar y evaluar ----
    vit::TrainConfig train_cfg;
    train_cfg.epochs       = 100;
    train_cfg.batch_size   = 128;
    train_cfg.lr           = 1e-3f;    // learning rate inicial
    train_cfg.min_lr       = 1e-4f;    // valor final del coseno
    train_cfg.warmup_ratio = 0.0f;     // sin warmup: decae desde el primer paso
    train_cfg.use_adam     = true;     // false = SGD plano, sin scheduler

    train_cfg.adam.beta1        = 0.9f;
    train_cfg.adam.beta2        = 0.999f;
    train_cfg.adam.weight_decay = 0.0f;

    std::printf("Entrenamiento:\n");
    std::printf("  Optimizador: %s\n", train_cfg.use_adam ? "Adam" : "SGD");
    std::printf("  Epochs:      %d\n", train_cfg.epochs);
    std::printf("  Batch size:  %d\n", train_cfg.batch_size);
    std::printf("  LR inicial:  %g\n", train_cfg.lr);
    if (train_cfg.use_adam) {
        std::printf("  LR final:    %g\n", train_cfg.min_lr);
        if (train_cfg.warmup_ratio > 0.0f) {
            std::printf("  Scheduler:   warmup lineal %.0f%% + coseno\n",
                        train_cfg.warmup_ratio * 100.0f);
        } else {
            std::printf("  Scheduler:   coseno (sin warmup)\n");
        }
    }
    std::printf("\n");

    try {
        vit::train_vit(
            train_img_ptr, train_lbl_ptr, n_train,
            val_img_ptr,   val_lbl_ptr,   n_val,
            test_data.images.data(), test_data.labels.data(), test_data.n_samples,
            weights, config,
            train_cfg);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error durante entrenamiento: %s\n", e.what());
        vit::vit_free_weights(weights, config);
        return 1;
    }

    // ---- 5. Limpiar ----
    vit::vit_free_weights(weights, config);
    std::printf("\nMemoria liberada. Fin.\n");

    return 0;
}
