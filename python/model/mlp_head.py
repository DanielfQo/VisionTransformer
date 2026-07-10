import numpy as np

from layers.layer_norm import LayerNorm


class MLPHead:
    def __init__(self, d_model, n_classes, seed=None):
        self.norm = LayerNorm(d_model)

        rng = np.random.default_rng(seed)
        limit = np.sqrt(6.0 / (d_model + n_classes))
        self.weight = rng.uniform(-limit, limit, (d_model, n_classes))
        self.bias = np.zeros((n_classes,))

    def forward(self, x):
        x = np.asarray(x, dtype=float)

        # tomamos solo el token cls (pos = 0)
        cls = x[:, 0:1, :]
        cls = np.asarray(self.norm(cls), dtype=float)
        cls = cls[:, 0, :]

        return np.matmul(cls, self.weight) + self.bias

    def __call__(self, x):
        return self.forward(x)


ClassifierHead = MLPHead
