# Vision Transformer en CUDA (MNIST)

Implementación autocontenida y optimizada de un **Vision Transformer (ViT)** en CUDA para clasificación de imágenes (**MNIST**).

---

## ⚡ Requisitos

- GPU NVIDIA
- CUDA Toolkit (11.x, 12.x o 13.x)
- CMake 3.20+
- MSVC (Windows) o GCC/G++ (Linux)

---

## 🚀 Cómo Compilar y Ejecutar

### **1. Compilar el Proyecto (Windows PowerShell)**

```powershell
# Entrar a la carpeta cuda
cd cuda

# Configurar el proyecto con CMake
cmd.exe /c "call ""C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"" && cmake -G ""Ninja"" -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86"

# Compilar en modo Release
cmd.exe /c "call ""C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"" && cmake --build build --config Release"
```

### **2. Ejecutar el ViT en MNIST**

```powershell
.\build\vit_train.exe
```

### **4. Recompilar y Ejecutar tras hacer cambios**

```powershell
cmd.exe /c "call ""C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"" && cmake --build build --config Release" ; .\build\vit_train.exe
```

### **5. Limpiar el Build (Clean Build)**

Si deseas eliminar los binarios y archivos temporales para compilar completamente desde cero:

#### **En Windows (PowerShell):**
```powershell
# Opcion A: Usando CMake clean target
cmake --build build --target clean

# Opcion B: Borrar la carpeta build por completo
Remove-Item -Recurse -Force build
```

#### **En Linux / WSL:**
```bash
# Opcion A: Usando CMake clean target
cmake --build build --target clean

# Opcion B: Borrar la carpeta build por completo
rm -rf build
```

### **6. Compilar en Linux / WSL**

```bash
cd cuda
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build . --config Release
./vit_train
```

---

## 🧪 Pruebas Unitarias en GPU

Para ejecutar las pruebas unitarias CUDA de cada componente:

```powershell
.\build\test_attention.exe
.\build\test_encoder.exe
.\build\test_patch_embedding.exe
```

---

## 📁 Estructura del Código CUDA

- **`capa1_nucleo/`**:
  - `tensor.cu`: Clase `Tensor` multidimensional.
  - `cuda_utils.cuh`: Funciones de reducción rápida usando **Warp Shuffle** (`__shfl_down_sync`).
- **`capa2_componentes/`**:
  - `patch_embedding`: Divide la imagen 28x28 en parches de 7x7 y proyecta los tokens con embeddings de posición.
  - `attention`: Multi-Head Self-Attention optimizado con el patrón **FlashAttention** (online softmax).
  - `encoder`: LayerNorm y MLP (capas lineales + activación GELU).
  - `softmax`: Softmax CUDA optimizado con reducción en memoria compartida.
- **`capa3_modelo/`**:
  - `vit.cu`: Modelo Vision Transformer completo (patch embedding → bloques encoder → extracción token CLS → clasificación).
- **`capa4_entrenamiento/`**:
  - `mnist_loader`: Lector de archivos binarios IDX de MNIST.
  - `training`: Loss Cross-Entropy, actualización SGD y loop de entrenamiento/evaluación.
  - `main.cu`: Punto de entrada principal.
- **`tests/`**: Pruebas unitarias para verificar los componentes en la GPU.
