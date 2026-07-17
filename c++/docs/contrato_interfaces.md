# Contrato — la Capa 1 (Núcleo), congelada el día 1

En esta arquitectura el "contrato" no es un acuerdo de palabra: es la **capa 1** (`tensor.hpp` +
`module.hpp`). Se define entre todos el día 1 y **no se cambia sin avisar**. Todo lo demás cuelga
de aquí, por eso los 4 integrantes pueden trabajar en paralelo.

> Nota: aquí va solo la *forma* del contrato. El código real se escribe en `tensor.*` y `module.hpp`.

## `tensor.hpp` — el tipo de dato único

Clase `Tensor` que viaja por toda la red. Debe soportar como mínimo:
- formas 4D `(B, C, H, W)` (imágenes) y 3D/2D `(B, N, d)` (tokens/logits),
- acceso a dimensiones y a los datos (`float`),
- las operaciones que necesitan todos: `matmul`, `softmax`, `gelu`/`relu`, suma, transpuesta.

Definir el `Tensor` una sola vez **elimina** el riesgo de que cada quien invente su propio tipo.

## `module.hpp` — la interfaz base

```
class Module {
public:
    virtual Tensor forward(const Tensor& x) = 0;
    virtual ~Module() = default;
};
```

**Toda clase de la red hereda de `Module`.** Es lo que permite componer (el ViT llama `forward()`
de sus sub-módulos) y probar cada parte por separado.

## Contratos por componente (todos son `Module`)

| Componente | Archivo | `forward`: entrada → salida |
|---|---|---|
| `PatchEmbeddingPipeline` | `patch_embedding.*` (M1) | imagen `(B,C,H,W)` → tokens `(B, N+1, d_model)` |
| `MultiHeadSelfAttention` | `attention.*` (M2) | tokens `(B, N+1, d_model)` → tokens `(B, N+1, d_model)` |
| `TransformerEncoder` | `encoder.*` (M3) | tokens `(B, N+1, d_model)` → tokens `(B, N+1, d_model)` |
| `MLPHead` | `vit.*` (M4) | tokens `(B, N+1, d_model)` → logits `(B, n_classes)` |
| `VisionTransformer` | `vit.*` (M4) | imagen `(B,C,H,W)` → logits `(B, n_classes)` |

Hiperparámetros que cada componente recibe en su constructor: `in_channels`, `image_size`,
`patch_size`, `d_model`, `n_heads`, `n_layers`, `n_classes`, `mlp_hidden_dim`, `dropout`, `seed`.

## Reglas del equipo

1. **Congelar la capa 1 primero** (`tensor.hpp` + `module.hpp`). Cambiarla requiere avisar a todos.
2. Toda clase de la red **hereda de `Module`** y respeta la firma de `forward`.
3. Dependencias solo hacia abajo (capa N usa capa N-1). Nunca dependencia lateral por `.cpp`.
4. Si necesitas una parte ajena que no está lista → **stub inline** (un `Module` falso), no esperes.
