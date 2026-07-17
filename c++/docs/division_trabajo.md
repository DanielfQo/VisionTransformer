# División de trabajo — 4 integrantes

Cada integrante es dueño de **un par `.hpp` + `.cpp`** y mete ahí **todas las clases de su parte**.
**Cada clase hereda de `Module`** (capa 1) e implementa `forward()`. El "training" va aparte.

## Capa 1 — Núcleo (se hace entre todos el día 1, luego se congela)

| Archivo | Qué contiene |
|---|---|
| `tensor.hpp` / `.cpp` | clase `Tensor` (el dato único) + ops: `matmul`, `softmax`, `gelu`/`relu`, suma, transpuesta. |
| `module.hpp` | interfaz base `Module { virtual Tensor forward(const Tensor&) = 0; }`. |

Todos los componentes de abajo dependen **solo** de esta capa. Ver `contrato_interfaces.md`.

---

## Miembro 1 — `patch_embedding.hpp` / `.cpp`  (capa 2)

Imagen → secuencia de tokens. Clases dentro del archivo (cada una hereda de `Module`):

| Clase | Qué hace |
|---|---|
| `PatchEmbedding` | parte la imagen `(B,C,H,W)` en parches de `patch_size`, aplana cada parche y lo proyecta a `d_model`. Salida `(B, N, d_model)`, `N=(H/patch)²`. |
| `CLSToken` | vector aprendible `d_model` que se antepone a cada secuencia. `(B,N,d)` → `(B,N+1,d)`. |
| `PositionEmbedding` | tabla `(N+1, d_model)` que se suma token a token. |
| `PatchEmbeddingPipeline` | encadena `patch → cls → pos`. Valida `image_size % patch_size == 0`. Es lo que usa el ViT. |

- **Entrada:** imagen `(B,C,H,W)`. **Salida:** tokens `(B, N+1, d_model)`.
- **Depende de:** solo la capa 1 (`Tensor` + `Module`). Totalmente independiente.
- Referencia Python: `python/patch_embedding/`.

## Miembro 2 — `attention.hpp` / `.cpp`  (capa 2)

Self-attention multi-cabeza (`MultiHeadSelfAttention` hereda de `Module`):

| Elemento | Qué hace |
|---|---|
| `scaled_dot_product_attention(Q,K,V,mask)` | `softmax(Q·Kᵀ/√d_k)·V`. Devuelve salida y (opcional) pesos. |
| `MultiHeadSelfAttention` | valida `d_model % n_heads == 0`, separa en cabezas, aplica la función, une cabezas. |

- **Entrada/Salida:** tokens `(B, N+1, d_model)`.
- **Depende de:** solo la capa 1. Independiente — prueba con tokens aleatorios.
- Referencia Python: `python/attention/`.

## Miembro 3 — `encoder.hpp` / `.cpp`  (capa 2)

Bloque transformer y la pila de bloques (cada clase hereda de `Module`):

| Clase | Qué hace |
|---|---|
| `LayerNorm` | `(x - media)/√(var+eps)`, luego escala `γ` y desplaza `β`. |
| `MLP` | `Linear(d_model→hidden) → GELU/ReLU → Linear(hidden→d_model)`, dropout opcional. |
| `TransformerEncoderBlock` | `x + Attention(norm1(x))`, luego `x + MLP(norm2(x))` (residuales). |
| `TransformerEncoder` | apila `n_layers` bloques y pasa `x` por todos en orden. |

- **Entrada/Salida:** tokens `(B, N+1, d_model)`.
- **Depende de:** capa 1 + el **header** `attention.hpp` (M2). Para probar el encoder solo, compone
  un `Module` falso de attention (stub inline que devuelve la entrada).
- Referencia Python: `python/layers/`, `python/model/encoder_block.py`, `encoder.py`.

## Miembro 4 — `vit.hpp` / `.cpp`  (capa 3)

Cabeza de clasificación + ensamblado del modelo (Composite; ambas heredan de `Module`):

| Clase | Qué hace |
|---|---|
| `MLPHead` | toma **solo el token CLS** (posición 0) y lo proyecta a `n_classes` logits. `(B,N+1,d)` → `(B, n_classes)`. |
| `VisionTransformer` | conecta `PatchEmbeddingPipeline → TransformerEncoder → MLPHead`. Incluye los headers de 1, 2, 3. |

- **Entrada del ViT:** imagen `(B,C,H,W)`. **Salida:** logits `(B, n_classes)`.
- **Depende de:** capa 1 + los **headers** de 1, 2, 3. Es el que integra (compone los `Module`).
  Prueba con `Module` falsos inline hasta que existan las implementaciones reales.
- Referencia Python: `python/model/mlp_head.py`, `python/model/vit.py`.

---

## Aparte — `training.hpp` / `.cpp` + `main.cpp`  (capa 4)

No es de la división de arquitectura.

| Elemento | Archivo | Qué hace |
|---|---|---|
| `cross_entropy` | `training.*` | loss entre logits y etiquetas. |
| `accuracy` | `training.*` | predicciones correctas / total. |
| `DataLoader` | `training.*` | carga MNIST (de `data/`) y entrega batches. |
| `Trainer` | `training.*` | bucle de épocas: forward, loss, métrica, (backward si aplica). |
| `main()` | `main.cpp` | arma el `VisionTransformer`, carga datos y corre entrenamiento/evaluación. |

Referencia Python: `python/training/`, `python/tests/entrenar_mnist.py`.
