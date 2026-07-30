
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np

from patch_embedding.patch_embedding import PatchEmbedding
from patch_embedding.cls_token import CLSToken
from patch_embedding.position_embedding import PositionEmbedding
from patch_embedding.pipeline import PatchEmbeddingPipeline


# ─────────────────────────────────────────────────────────────────────────────
# Helpers / constantes compartidas
# ─────────────────────────────────────────────────────────────────────────────

# Variantes de ViT: (nombre, image_size, patch_size, d_model, in_channels)
VIT_VARIANTS = [
    ("ViT-Ti",      224, 16, 192,  3),
    ("ViT-S",       224, 16, 384,  3),
    ("ViT-B",       224, 16, 768,  3),
    ("ViT-L",       224, 16, 1024, 3),
    ("ViT-B/16-384",384, 16, 768,  3),
    ("ViT-B/32",    224, 32, 768,  3),
]


def n_patches(image_size: int, patch_size: int) -> int:
    return (image_size // patch_size) ** 2


def seq_len(image_size: int, patch_size: int) -> int:
    return 1 + n_patches(image_size, patch_size)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(
    params=[pytest.param(v, id=v[0]) for v in VIT_VARIANTS]
)
def vit_params(request):
    _, image_size, patch_size, d_model, in_channels = request.param
    return image_size, patch_size, d_model, in_channels


@pytest.fixture
def base():
    return dict(in_channels=3, image_size=224, patch_size=16, d_model=768)


@pytest.fixture
def batch_images(vit_params):
    image_size, patch_size, d_model, in_channels = vit_params
    return np.random.randn(4, in_channels, image_size, image_size).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Tests de PatchEmbedding
# ─────────────────────────────────────────────────────────────────────────────

class TestPatchEmbedding:
    def test_output_shape(self, vit_params, batch_images):
        image_size, patch_size, d_model, in_channels = vit_params
        B = batch_images.shape[0]
        N = n_patches(image_size, patch_size)

        model = PatchEmbedding(in_channels, patch_size, d_model)
        out = model(batch_images)

        assert out.shape == (B, N, d_model), (
            f"Esperado ({B}, {N}, {d_model}), obtenido {out.shape}"
        )

    def test_output_dtype(self, base):
        model = PatchEmbedding(**{k: base[k] for k in ("in_channels", "patch_size", "d_model")})
        x = np.random.randn(2, base["in_channels"], base["image_size"], base["image_size"]).astype(np.float32)
        assert np.issubdtype(model(x).dtype, np.floating)

    def test_batch_independence(self, base):
        model = PatchEmbedding(base["in_channels"], base["patch_size"], base["d_model"])
        images = np.random.randn(4, base["in_channels"], base["image_size"], base["image_size"]).astype(np.float32)

        out_batch  = model(images)
        out_single = np.concatenate([model(images[i:i+1]) for i in range(4)], axis=0)

        assert np.allclose(out_batch, out_single, atol=1e-5)

    def test_different_batch_sizes(self, base):
        model = PatchEmbedding(base["in_channels"], base["patch_size"], base["d_model"])
        N = n_patches(base["image_size"], base["patch_size"])
        for B in [1, 2, 8, 16]:
            x = np.random.randn(B, base["in_channels"], base["image_size"], base["image_size"]).astype(np.float32)
            assert model(x).shape == (B, N, base["d_model"])


# ─────────────────────────────────────────────────────────────────────────────
# Tests de CLSToken
# ─────────────────────────────────────────────────────────────────────────────

class TestCLSToken:
    def test_output_shape(self, vit_params):
        image_size, patch_size, d_model, _ = vit_params
        B = 4
        N = n_patches(image_size, patch_size)
        x = np.random.randn(B, N, d_model).astype(np.float32)

        out = CLSToken(d_model)(x)

        assert out.shape == (B, N + 1, d_model), (
            f"Esperado ({B}, {N+1}, {d_model}), obtenido {out.shape}"
        )

    def test_patch_tokens_unchanged(self, base):
        B, N, D = 2, n_patches(base["image_size"], base["patch_size"]), base["d_model"]
        x = np.random.randn(B, N, D).astype(np.float32)
        out = CLSToken(D)(x)

        assert np.array_equal(out[:, 1:, :], x)

    def test_cls_shared_across_batch(self, base):
        N, D = n_patches(base["image_size"], base["patch_size"]), base["d_model"]
        x = np.random.randn(4, N, D).astype(np.float32)
        model = CLSToken(D)
        out = model(x)

        cls_tokens = out[:, 0, :]  # (B, D)
        assert np.allclose(cls_tokens[0], cls_tokens[1])
        assert np.allclose(cls_tokens[0], cls_tokens[-1])


# ─────────────────────────────────────────────────────────────────────────────
# Tests de PositionEmbedding
# ─────────────────────────────────────────────────────────────────────────────

class TestPositionEmbedding:
    def test_output_shape(self, vit_params):
        image_size, patch_size, d_model, _ = vit_params
        B = 4
        S = seq_len(image_size, patch_size)
        x = np.random.randn(B, S, d_model).astype(np.float32)

        out = PositionEmbedding(S, d_model)(x)

        assert out.shape == (B, S, d_model)

    def test_adds_positional_info(self, base):
        S, D = seq_len(base["image_size"], base["patch_size"]), base["d_model"]
        x = np.random.randn(2, S, D).astype(np.float32)
        model = PositionEmbedding(S, D)
        model.pos_embedding.fill(1.0)

        assert not np.array_equal(model(x), x)

    def test_pos_embedding_shape(self, vit_params):
        image_size, patch_size, d_model, _ = vit_params
        S = seq_len(image_size, patch_size)
        model = PositionEmbedding(S, d_model)
        assert model.pos_embedding.shape == (1, S, d_model)


# ─────────────────────────────────────────────────────────────────────────────
# Tests de PatchEmbeddingPipeline (integración)
# ─────────────────────────────────────────────────────────────────────────────

class TestPatchEmbeddingPipeline:
    def test_output_shape(self, vit_params, batch_images):
        image_size, patch_size, d_model, in_channels = vit_params
        B = batch_images.shape[0]
        S = seq_len(image_size, patch_size)

        model = PatchEmbeddingPipeline(in_channels, image_size, patch_size, d_model)
        out = model(batch_images)

        assert out.shape == (B, S, d_model), (
            f"Esperado ({B}, {S}, {d_model}), obtenido {out.shape}"
        )

    def test_cls_token_at_position_zero(self, base):
        B, D = 2, base["d_model"]
        model = PatchEmbeddingPipeline(**base)

        model.cls_token.cls_token.fill(99.0)
        model.pos_embed.pos_embedding.fill(0.0)

        images = np.random.randn(B, base["in_channels"], base["image_size"], base["image_size"]).astype(np.float32)
        out = model(images)

        assert np.allclose(out[:, 0, :], np.full((B, D), 99.0))

    def test_all_variants_smoke(self):
        for name, image_size, patch_size, d_model, in_channels in VIT_VARIANTS:
            model = PatchEmbeddingPipeline(in_channels, image_size, patch_size, d_model)
            x = np.random.randn(1, in_channels, image_size, image_size).astype(np.float32)
            out = model(x)
            S = seq_len(image_size, patch_size)
            assert out.shape == (1, S, d_model), (
                f"Fallo en {name}: esperado (1, {S}, {d_model}), obtenido {out.shape}"
            )

    def test_invalid_image_size(self, base):
        with pytest.raises(AssertionError):
            PatchEmbeddingPipeline(
                in_channels=3, image_size=225, patch_size=16, d_model=768
            )

    def test_output_dtype(self, base):
        model = PatchEmbeddingPipeline(**base)
        x = np.random.randn(2, base["in_channels"], base["image_size"], base["image_size"]).astype(np.float32)
        assert np.issubdtype(model(x).dtype, np.floating)
