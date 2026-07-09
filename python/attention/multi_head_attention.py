from .scaled_dot_product_attention import scaled_dot_product_attention


class MultiHeadSelfAttention:
    """Self-attention multi-cabeza hecha a mano."""

    def __init__(self, d_model, n_heads):
        if d_model % n_heads != 0:
            raise ValueError("d_model debe ser divisible entre n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.dim_cabeza = d_model // n_heads

    def separar_cabezas(self, x):
        salida = []

        for batch in x:
            batch_heads = [[] for _ in range(self.n_heads)]

            for token in batch:
                for h in range(self.n_heads):
                    ini = h * self.dim_cabeza
                    fin = ini + self.dim_cabeza
                    batch_heads[h].append(token[ini:fin])

            salida.append(batch_heads)

        return salida

    def unir_cabezas(self, x):
        salida = []

        for batch in x:
            tokens = len(batch[0])
            batch_salida = []

            for t in range(tokens):
                token = []
                for h in range(self.n_heads):
                    token += batch[h][t]
                batch_salida.append(token)

            salida.append(batch_salida)

        return salida

    def forward(self, x, mask=None, devolver_pesos=False):
        query = self.separar_cabezas(x)
        key = self.separar_cabezas(x)
        value = self.separar_cabezas(x)

        salida, pesos = scaled_dot_product_attention(query, key, value, mask)
        salida = self.unir_cabezas(salida)

        if devolver_pesos:
            return salida, pesos
        return salida

    def __call__(self, x, mask=None, devolver_pesos=False):
        return self.forward(x, mask, devolver_pesos)


MultiHeadAttention = MultiHeadSelfAttention
