import numpy as np

from attention.multi_head_attention import MultiHeadSelfAttention
from layers.layer_norm import LayerNorm
from layers.mlp import MLP


class TransformerEncoderBlock:
    def __init__(self, d_model, n_heads, mlp_hidden_dim=None, dropout=0.0, seed=None):
        self.d_model = d_model
        self.n_heads = n_heads
        self.norm1 = LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(d_model, n_heads)
        self.norm2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, hidden_dim=mlp_hidden_dim, dropout=dropout, seed=seed)

    def _mantener_tipo(self, salida, entrada_es_lista):
        if entrada_es_lista:
            return salida.tolist()
        return salida

    def forward(self, x, mask=None, training=False, devolver_pesos=False):
        entrada_es_lista = isinstance(x, list)
        x_np = np.asarray(x, dtype=float)

        normalizado = self.norm1(x_np.tolist())
        if devolver_pesos:
            attn_out, pesos = self.attention(normalizado, mask=mask, devolver_pesos=True)
        else:
            attn_out = self.attention(normalizado, mask=mask)
            pesos = None

        x_np = x_np + np.asarray(attn_out, dtype=float)
        normalizado = self.norm2(x_np.tolist())
        mlp_out = self.mlp(normalizado, training=training)
        salida = x_np + np.asarray(mlp_out, dtype=float)

        salida = self._mantener_tipo(salida, entrada_es_lista)
        if devolver_pesos:
            return salida, pesos
        return salida

    def __call__(self, x, mask=None, training=False, devolver_pesos=False):
        return self.forward(x, mask=mask, training=training, devolver_pesos=devolver_pesos)


EncoderBlock = TransformerEncoderBlock
