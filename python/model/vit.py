from patch_embedding import PatchEmbeddingPipeline
from model.encoder import TransformerEncoder
from model.mlp_head import MLPHead

# vit: patch embedding -> encoder -> mlp clasifica


class VisionTransformer:
    def __init__(self, in_channels, image_size, patch_size, d_model,
                 n_heads, n_layers, n_classes,
                 mlp_hidden_dim=None, dropout=0.0, seed=None):
        self.patch = PatchEmbeddingPipeline(in_channels, image_size, patch_size, d_model)
        self.encoder = TransformerEncoder(d_model, n_heads, n_layers,
                                          mlp_hidden_dim=mlp_hidden_dim,
                                          dropout=dropout, seed=seed)
        self.head = MLPHead(d_model, n_classes, seed=seed)

    def forward(self, x, training=False, devolver_pesos=False):
        x = self.patch(x)  # (B, N+1, d_model)

        if devolver_pesos:
            x, pesos = self.encoder(x, training=training, devolver_pesos=True)
        else:
            x = self.encoder(x, training=training)

        logits = self.head(x)  # (B, n_classes)

        if devolver_pesos:
            return logits, pesos
        return logits

    def __call__(self, x, training=False, devolver_pesos=False):
        return self.forward(x, training=training, devolver_pesos=devolver_pesos)


ViT = VisionTransformer
