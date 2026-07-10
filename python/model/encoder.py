from model.encoder_block import TransformerEncoderBlock

class TransformerEncoder: 
    
    # apila n bloques de encoder
    def __init__(self, d_model, n_heads, n_layers,
                 mlp_hidden_dim=None, dropout=0.0, seed=None):
        self.d_model = d_model
        self.n_layers = n_layers

        self.bloques = []
        for i in range(n_layers):
            seed_bloque = None if seed is None else seed + i
            bloque = TransformerEncoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                mlp_hidden_dim=mlp_hidden_dim,
                dropout=dropout,
                seed=seed_bloque,
            )
            self.bloques.append(bloque)

    def forward(self, x, mask=None, training=False, devolver_pesos=False):
        pesos_por_capa = []

        # pasamos x por cada bloque en orden
        for bloque in self.bloques:
            if devolver_pesos:
                x, pesos = bloque(x, mask=mask, training=training,
                                  devolver_pesos=True)
                pesos_por_capa.append(pesos)
            else:
                x = bloque(x, mask=mask, training=training)

        if devolver_pesos:
            return x, pesos_por_capa
        return x

    def __call__(self, x, mask=None, training=False, devolver_pesos=False):
        return self.forward(x, mask=mask, training=training,
                            devolver_pesos=devolver_pesos)


Encoder = TransformerEncoder
