
import numpy as np


class PositionEmbedding:

    def __init__(self, seq_len: int, d_model: int) -> None:

        self.pos_embedding = np.random.normal(0, 0.02, (1, seq_len, d_model))

    def forward(self, x: np.ndarray) -> np.ndarray:

        return x + self.pos_embedding

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)