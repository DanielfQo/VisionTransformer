# Arquitectura — por capas + patrón `Module`

Mismo diseño que la versión de `python/` y que los frameworks reales (PyTorch `nn.Module`).
El dato que viaja entre todo es la clase `Tensor` (capa 1).

## El patrón central: `Module`

```
// module.hpp  (capa 1, congelada el día 1)
class Module {
public:
    virtual Tensor forward(const Tensor& x) = 0;   // interfaz uniforme
    virtual ~Module() = default;
};
```

Todo componente de la red **hereda de `Module`** e implementa `forward()`. Así:
- son intercambiables (Strategy),
- son componibles: un `Module` puede contener otros y llamarlos en cadena (Composite),
- se prueban por separado a través de la misma interfaz.

## Las 4 capas

```
        imagen : Tensor (B, C, H, W)
              │
              ▼
┌───────────────────────────────┐  CAPA 2 · Componentes (ES-UN Module)
│ PatchEmbeddingPipeline  (M1)  │  patch → cls_token → position
└───────────────────────────────┘
              │ tokens (B, N+1, d_model)
              ▼
┌───────────────────────────────┐
│ TransformerEncoder      (M3)  │  apila n_layers EncoderBlock
│   cada bloque usa:            │
│     ┌───────────────────────┐ │
│     │ MultiHeadAttention(M2)│ │  ← también es un Module
│     └───────────────────────┘ │
└───────────────────────────────┘
              │ tokens (B, N+1, d_model)
              ▼
┌───────────────────────────────┐  CAPA 3 · Modelo
│ MLPHead                 (M4)  │  token CLS → logits
└───────────────────────────────┘
              │ logits (B, n_classes)
              ▼
┌───────────────────────────────┐  CAPA 4 · Entrenamiento
│ Trainer / main                │  loss, accuracy, MNIST, épocas
└───────────────────────────────┘

VisionTransformer (M4, capa 3) es un Module que COMPONE M1 → M3 → MLPHead.
```

## Regla de dependencias: solo hacia abajo

| Parte | Capa | Depende de |
|---|---|---|
| `tensor.*`, `module.hpp` | 1 · Núcleo | nada (es el fundamento) |
| `patch_embedding.*` (M1) | 2 | solo capa 1 |
| `attention.*` (M2) | 2 | solo capa 1 |
| `encoder.*` (M3) | 2 | capa 1 + header `attention.hpp` |
| `vit.*` (M4) | 3 | capa 1 + headers de M1, M2, M3 |
| `training.*`, `main.cpp` | 4 | capa 3 (usa `VisionTransformer`) |

- **Nunca hay dependencia lateral dentro de la capa 2.** M1, M2, M3 no se llaman entre sí como
  implementaciones; M3 solo incluye el *header* de M2 (contrato), no su `.cpp`.
- Como todos programan contra la capa 1 (congelada), los 4 compilan y prueban en paralelo.

## Integración

Componer los `Module` reales en `vit.cpp` es solo instanciar las clases y encadenarlas. Mientras
falte una parte, se compone un `Module` falso (stub inline) que cumple la interfaz `forward()` y
devuelve un `Tensor` con la forma correcta. No se toca la lógica de nadie.
