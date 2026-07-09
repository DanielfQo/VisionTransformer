import math


class LayerNorm:

    def __init__(self, d_model, eps=1e-5):
        self.eps = eps
        self.gamma = [1.0 for _ in range(d_model)]
        self.beta = [0.0 for _ in range(d_model)]

    def normalizar_vector(self, vector):
        media = sum(vector) / len(vector)

        varianza = 0.0
        for valor in vector:
            varianza += (valor - media) ** 2
        varianza /= len(vector)

        salida = []
        for i in range(len(vector)):
            valor = (vector[i] - media) / math.sqrt(varianza + self.eps)
            salida.append(valor * self.gamma[i] + self.beta[i])

        return salida

    def forward(self, x):
        salida = []

        for batch in x:
            batch_salida = []
            for token in batch:
                batch_salida.append(self.normalizar_vector(token))
            salida.append(batch_salida)

        return salida

    def __call__(self, x):
        return self.forward(x)
