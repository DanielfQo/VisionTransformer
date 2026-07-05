import math


def producto_punto(a, b):
    total = 0.0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total


def softmax(valores):
    mayor = max(valores)
    exp_valores = []
    suma = 0.0

    for valor in valores:
        exp_valor = math.exp(valor - mayor)
        exp_valores.append(exp_valor)
        suma += exp_valor

    return [valor / suma for valor in exp_valores]


def suma_ponderada(pesos, vectores):
    dim = len(vectores[0])
    salida = [0.0 for _ in range(dim)]

    for i in range(len(pesos)):
        for j in range(dim):
            salida[j] += pesos[i] * vectores[i][j]

    return salida


def usar_mask(mask, batch, cabeza, fila, col):
    if mask is None:
        return True

    actual = mask
    for indice in (batch, cabeza, fila, col):
        if isinstance(actual, list) and actual and isinstance(actual[0], list):
            actual = actual[indice]
        else:
            break

    return bool(actual)


def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Attention hecha a mano.
    Formato esperado: [batch][cabezas][tokens][dim].
    """
    dim = len(query[0][0][0])
    escala = math.sqrt(dim)
    salida_total = []
    pesos_total = []

    for b in range(len(query)):
        salida_batch = []
        pesos_batch = []

        for h in range(len(query[b])):
            salida_cabeza = []
            pesos_cabeza = []

            for i in range(len(query[b][h])):
                puntajes = []

                for j in range(len(key[b][h])):
                    puntaje = producto_punto(query[b][h][i], key[b][h][j]) / escala
                    if not usar_mask(mask, b, h, i, j):
                        puntaje = -1e9
                    puntajes.append(puntaje)

                pesos = softmax(puntajes)
                salida = suma_ponderada(pesos, value[b][h])

                pesos_cabeza.append(pesos)
                salida_cabeza.append(salida)

            pesos_batch.append(pesos_cabeza)
            salida_batch.append(salida_cabeza)

        pesos_total.append(pesos_batch)
        salida_total.append(salida_batch)

    return salida_total, pesos_total


class ScaledDotProductAttention:
    def forward(self, query, key, value, mask=None):
        return scaled_dot_product_attention(query, key, value, mask)

    def __call__(self, query, key, value, mask=None):
        return self.forward(query, key, value, mask)
