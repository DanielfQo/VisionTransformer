import numpy as np


class MLP:
    def __init__(self, d_model, hidden_dim=None, dropout=0.0, seed=None):
        self.d_model = d_model
        self.hidden_dim = hidden_dim or d_model * 4
        self.dropout = dropout
        self.rng = np.random.default_rng(seed)

        limit_1 = np.sqrt(6.0 / (d_model + self.hidden_dim))
        limit_2 = np.sqrt(6.0 / (self.hidden_dim + d_model))
        self.w1 = self.rng.uniform(-limit_1, limit_1, (d_model, self.hidden_dim))
        self.b1 = np.zeros((self.hidden_dim,))
        self.w2 = self.rng.uniform(-limit_2, limit_2, (self.hidden_dim, d_model))
        self.b2 = np.zeros((d_model,))

    def gelu(self, x):
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))

    def _dropout(self, x, training):
        if not training or self.dropout <= 0.0:
            return x

        keep_prob = 1.0 - self.dropout
        mask = self.rng.binomial(1, keep_prob, size=x.shape)
        return (x * mask) / keep_prob

    def forward(self, x, training=False):
        entrada_lista = isinstance(x, list)
        x_np = np.asarray(x, dtype=float)

        hidden = self.gelu(np.matmul(x_np, self.w1) + self.b1)
        hidden = self._dropout(hidden, training)
        salida = np.matmul(hidden, self.w2) + self.b2
        salida = self._dropout(salida, training)

        if entrada_lista:
            return salida.tolist()
        return salida

    def __call__(self, x, training=False):
        return self.forward(x, training)
