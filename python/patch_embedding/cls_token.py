"""
El token [CLS] es un token de dimension d_model que se concatena como primer
elemento de la secuencia en cada elemento del batch.


in : (B, N, d_model)   secuencia de embeddings de parche
out  : (B, N+1, d_model)   CLS token como primer elemento
"""

import numpy as np


class CLSToken:

    def __init__(self, d_model: int) -> None:

        self.cls_token = np.zeros((1, 1, d_model)) #TODO: inicializar con dist normal o uniforme

    def forward(self, x: np.ndarray) -> np.ndarray:

        B = x.shape[0]

        # Expandimos el token CLS en la dimension batch
        cls = np.repeat(self.cls_token, B, axis=0)

        return np.concatenate([cls, x], axis=1)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)