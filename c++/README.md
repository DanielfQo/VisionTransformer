# Vision Transformer — Implementación en C++

Reimplementación del ViT (ya hecho en `python/`) en C++ **desde cero**, sin librerías de
deep learning. Repartido entre **4 integrantes**.

## Arquitectura: por capas + patrón `Module` (Composite)

Misma idea que PyTorch (`nn.Module`) o Keras (`Layer`): **todo componente de la red es un
`Module`** con una interfaz uniforme `forward(entrada) → salida`. El ViT no es un caso especial;
es un `Module` que **compone** otros `Module`.

```
┌─────────────────────────────────────────────────────┐
│ CAPA 4 · Entrenamiento   training.*  main.cpp         │  orquesta: loss, datos, épocas
├─────────────────────────────────────────────────────┤
│ CAPA 3 · Modelo          vit.*                        │  Composite: arma la red completa
├─────────────────────────────────────────────────────┤
│ CAPA 2 · Componentes     patch_embedding.* (M1)       │  cada uno ES-UN Module
│                          attention.*       (M2)       │
│                          encoder.*         (M3)       │
├─────────────────────────────────────────────────────┤
│ CAPA 1 · Núcleo          tensor.*   module.hpp        │  tipo de dato + interfaz base
└─────────────────────────────────────────────────────┘
```

### Por qué esta arquitectura

- **`module.hpp` — interfaz `Module`.** Clase base abstracta con `virtual Tensor forward(...) = 0`.
  Es el *contrato* (patrón Composite/Strategy): todos los componentes son intercambiables y
  componibles. El ViT guarda punteros a `Module` y los llama en cadena.
- **`tensor.*` — capa de fundamento.** El único tipo de dato que viaja por toda la red. Resuelve
  de raíz el "todos deben usar el mismo tipo": ya no es un acuerdo de palabra, es una clase.
- **Dependencia hacia abajo, nunca lateral.** Cada capa solo depende de la de abajo. Los 4
  componentes (capa 2) dependen **solo de la capa 1** (`Tensor` + `Module`), no unos de otros.
- **Open/Closed:** se agregan o cambian componentes sin tocar los demás, mientras respeten `Module`.

## Estructura (una carpeta por capa)

```
c++/
├── capa1_nucleo/                 # CAPA 1 — fundamento (se congela el día 1)
│   ├── tensor.hpp / .cpp         #   clase Tensor + ops (matmul, softmax, gelu, ...)
│   └── module.hpp                #   interfaz base Module { forward() }
│
├── capa2_componentes/            # CAPA 2 — cada componente ES-UN Module
│   ├── patch_embedding.hpp/.cpp  #   MIEMBRO 1 — imagen → tokens
│   ├── attention.hpp     /.cpp   #   MIEMBRO 2 — self-attention multi-cabeza
│   └── encoder.hpp       /.cpp   #   MIEMBRO 3 — layer_norm, mlp, encoder_block, encoder
│
├── capa3_modelo/                 # CAPA 3 — Composite
│   └── vit.hpp           /.cpp   #   MIEMBRO 4 — mlp_head + VisionTransformer
│
├── capa4_entrenamiento/          # CAPA 4 — orquestación
│   ├── training.hpp      /.cpp   #   loss, métricas, data loader, trainer
│   └── main.cpp                  #   punto de entrada: arma el ViT y entrena
│
├── data/                         # archivos de MNIST
├── CMakeLists.txt
└── docs/
    ├── arquitectura.md           # capas, patrón Module, flujo de datos
    ├── division_trabajo.md       # qué clase va en cada archivo
    └── contrato_interfaces.md    # la capa 1 congelada (Tensor + Module)
```

Cada integrante mete **todas las clases de su parte en su único `.hpp`/`.cpp`**, y **cada una
hereda de `Module`**. Una capa solo incluye headers de la capa de abajo (nunca lateral).

## Qué tiene que hacer cada integrante

Todas las clases heredan de `Module` e implementan `forward(Tensor) → Tensor`.

### 🟦 Miembro 1 — `capa2_componentes/patch_embedding.*` (imagen → tokens)

Convierte la imagen en una secuencia de tokens. Clases a implementar:
- `PatchEmbedding` — parte la imagen `(B,C,H,W)` en parches de `patch_size`, aplana cada parche y lo proyecta a `d_model`. Salida `(B, N, d_model)`, con `N=(H/patch)²`.
- `CLSToken` — vector aprendible de tamaño `d_model` que se antepone a cada secuencia. `(B,N,d) → (B,N+1,d)`.
- `PositionEmbedding` — tabla `(N+1, d_model)` que se suma token a token.
- `PatchEmbeddingPipeline` — encadena `patch → cls → pos`. Valida `image_size % patch_size == 0`.

**Depende de:** solo la capa 1. Independiente. → Referencia Python: `python/patch_embedding/`.

### 🟩 Miembro 2 — `capa2_componentes/attention.*` (tokens → tokens)

Self-attention multi-cabeza. A implementar:
- `scaled_dot_product_attention(Q,K,V,mask)` — `softmax(Q·Kᵀ/√d_k)·V`. Devuelve salida y (opcional) pesos.
- `MultiHeadSelfAttention` — valida `d_model % n_heads == 0`; separa en `n_heads` cabezas, aplica la función, une las cabezas.

**Depende de:** solo la capa 1. Independiente. → Referencia Python: `python/attention/`.

### 🟨 Miembro 3 — `capa2_componentes/encoder.*` (tokens → tokens)

Bloque transformer y la pila de bloques. Clases:
- `LayerNorm` — `(x - media)/√(var+eps)`, luego escala `γ` y desplaza `β`.
- `MLP` — `Linear(d_model→hidden) → GELU/ReLU → Linear(hidden→d_model)`, con dropout opcional.
- `TransformerEncoderBlock` — `x + Attention(norm1(x))`, luego `x + MLP(norm2(x))` (residuales).
- `TransformerEncoder` — apila `n_layers` bloques y pasa `x` por todos en orden.

**Depende de:** capa 1 + header `attention.hpp` (M2). Mientras M2 no esté, prueba con un `Module` falso de attention (stub inline). → Referencia Python: `python/layers/`, `python/model/encoder_block.py`, `encoder.py`.

### 🟥 Miembro 4 — `capa3_modelo/vit.*` (imagen → logits, integra todo)

Cabeza de clasificación + ensamblado del modelo (Composite). Clases:
- `MLPHead` — toma **solo el token CLS** (posición 0) y lo proyecta a `n_classes` logits. `(B,N+1,d) → (B, n_classes)`.
- `VisionTransformer` — compone `PatchEmbeddingPipeline → TransformerEncoder → MLPHead`.

**Depende de:** capa 1 + headers de M1, M2, M3. Es el que integra; prueba con `Module` falsos inline hasta que existan los reales. → Referencia Python: `python/model/mlp_head.py`, `python/model/vit.py`.

### ⬜ Aparte — `capa4_entrenamiento/` (no cuenta en la división de los 4)

- `cross_entropy`, `accuracy` (en `training.*`) — loss y métrica.
- `DataLoader` (en `training.*`) — carga MNIST desde `data/` y entrega batches.
- `Trainer` (en `training.*`) — bucle de épocas: forward, loss, métrica, (backward si aplica).
- `main()` (en `main.cpp`) — arma el `VisionTransformer`, carga datos y corre entrenamiento/evaluación.

→ Referencia Python: `python/training/`, `python/tests/entrenar_mnist.py`.

## Cómo trabajar sin depender del avance del resto

1. **Día 1 (juntos):** definir y **congelar la capa 1** (`tensor.hpp` + `module.hpp`). Es el
   contrato del que cuelga todo lo demás. Ver `docs/contrato_interfaces.md`.
2. **En paralelo:** cada quien implementa sus clases heredando de `Module`, contra la capa 1.
   Compila aunque el resto esté vacío. Prueba con un `main`/stub propio.
3. **Integración:** `vit.cpp` compone los `Module` reales; `main.cpp` + `training.*` entrenan.

## Orden de trabajo sugerido

- M1, M2, M3, M4 avanzan a la vez sobre la capa 1 congelada.
- El ViT (M4) puede probarse desde el día 1 componiendo `Module` falsos (stubs inline) que
  cumplen la interfaz, y cambiándolos por los reales al integrar.

Ver `docs/division_trabajo.md` para el detalle de qué clase va en cada archivo.

## Compilación de multi-head self-attention

Archivos C++: `capa2_componentes/attention.hpp` y
`capa2_componentes/attention.cpp`.

Archivos CUDA: `../cuda/capa2_componentes/attention.cuh` y
`../cuda/capa2_componentes/attention.cu`.

La entrada y salida tienen layout contiguo `[batch, tokens, d_model]`. Cada
cabeza trabaja con `head_dim = d_model / heads`. Esta version sigue el contrato
de la implementacion Python actual: usa los propios tokens como Q, K y V, sin
matrices de proyeccion aprendibles.

La mascara opcional tiene layout `[batch, heads, tokens, tokens]` y tipo
`uint8_t`: cero bloquea una posicion y cualquier otro valor la permite. Una fila
completamente bloqueada produce ceros.

Compilar y probar la referencia CPU:

```bash
c++ -std=c++17 -O2 \
  capa2_componentes/attention.cpp tests/test_attention.cpp \
  -o test_attention
./test_attention
```

Compilar el componente CUDA:

```bash
nvcc -std=c++17 -O2 -c \
  c++/capa2_componentes/attention.cpp cuda/capa2_componentes/attention.cu
```

`multi_head_self_attention_cuda` recibe punteros ya reservados en la GPU. El
buffer de salida requiere `batch * tokens * d_model` valores `float`; el buffer
opcional de pesos requiere `batch * heads * tokens * tokens`.
