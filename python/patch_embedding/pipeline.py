
import numpy as np

from .patch_embedding import PatchEmbedding
from .cls_token import CLSToken
from .position_embedding import PositionEmbedding


class PatchEmbeddingPipeline:

    def __init__(
        self,
        in_channels: int,
        image_size: int,
        patch_size: int,
        d_model: int,
    ) -> None:
        assert image_size % patch_size == 0, (
            f"image_size ({image_size}) debe ser divisible por patch_size ({patch_size})"
        )

        self.n_patches = (image_size // patch_size) ** 2
        self.seq_len   = 1 + self.n_patches

        self.patch_embed = PatchEmbedding(in_channels, patch_size, d_model)
        self.cls_token   = CLSToken(d_model)
        self.pos_embed   = PositionEmbedding(self.seq_len, d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:

        x = self.patch_embed(x)  # (B, N, d_model)
        x = self.cls_token(x)    # (B, N+1, d_model)
        x = self.pos_embed(x)    # (B, N+1, d_model)
        return x #secuencia de tokens

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
