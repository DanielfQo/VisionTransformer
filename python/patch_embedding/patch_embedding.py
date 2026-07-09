"""

Dado un tensor de imagen (B, C, H, W):
  1. Divide la imagen en parches cuadrados de tam patch_size
  2. Aplana cada parche
  3. Aplica una proyeccion lineal

"""

import numpy as np


class PatchEmbedding:

    def __init__(
        self,
        in_channels: int, #num de canales de la img
        patch_size: int, #tamaño del lado del parche
        d_model: int, #dim del embedding de salida
    ) -> None:
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.d_model = d_model
        
        # Dimensión plana de un parche: C * P * P
        self.patch_dim = in_channels * patch_size * patch_size

        # inicializacion xavier
        limit = np.sqrt(6.0 / (self.patch_dim + d_model))
        self.weight = np.random.uniform(-limit, limit, (self.patch_dim, d_model))
        self.bias = np.zeros((d_model,))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        in:
            x: shape (B, C, H, W).

        out:
            Array de forma (B, N, d_model).
        """
        B, C, H, W = x.shape
        P = self.patch_size

        assert H % P == 0 and W % P == 0, (
            f"La altura ({H}) y ancho ({W}) de la imagen deben ser divisibles por patch_size ({P})"
        )

        num_patches_h = H // P
        num_patches_w = W // P
        N = num_patches_h * num_patches_w


        x_reshaped = x.reshape(B, C, num_patches_h, P, num_patches_w, P)
        
        # Transponer a (B, num_patches_h, num_patches_w, C, P, P)
        x_transposed = x_reshaped.transpose(0, 2, 4, 1, 3, 5)
        
        # Aplanar cada parche: (B, N, C * P * P)
        x_flat = x_transposed.reshape(B, N, self.patch_dim)

        # proyeccion lineal
        out = np.matmul(x_flat, self.weight) + self.bias

        return out

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)