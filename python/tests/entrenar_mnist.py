import argparse
import os
import sys

import numpy as np
from torchvision import datasets, transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.vit import VisionTransformer


def cargar_datos(ruta, train, cantidad):
    datos = datasets.MNIST(
        root=ruta,
        train=train,
        download=True,
        transform=transforms.ToTensor(),
    )

    imagenes = []
    etiquetas = []

    for i in range(min(cantidad, len(datos))):
        imagen, etiqueta = datos[i]
        imagenes.append(imagen.numpy())
        etiquetas.append(etiqueta)

    return np.asarray(imagenes, dtype=float), np.asarray(etiquetas, dtype=int)


def lotes(x, y, tam, mezclar=True):
    indices = np.arange(len(x))

    if mezclar:
        np.random.shuffle(indices)

    for ini in range(0, len(indices), tam):
        sel = indices[ini:ini + tam]
        yield x[sel], y[sel]


def softmax(logits):
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)


def loss_accuracy(logits, y):
    probs = softmax(logits)
    loss = -np.mean(np.log(probs[np.arange(len(y)), y] + 1e-12))
    pred = np.argmax(probs, axis=1)
    acc = np.mean(pred == y)
    return loss, acc


def obtener_features(modelo, x):
    tokens = modelo.patch(x)
    tokens = modelo.encoder(tokens, training=False)
    cls = np.asarray(tokens, dtype=float)[:, 0:1, :]
    cls = np.asarray(modelo.head.norm(cls), dtype=float)[:, 0, :]
    return cls


def evaluar(modelo, x, y, tam):
    perdidas = []
    correctos = 0
    total = 0

    for xb, yb in lotes(x, y, tam, mezclar=False):
        feat = obtener_features(modelo, xb)
        logits = feat @ modelo.head.weight + modelo.head.bias
        loss, acc = loss_accuracy(logits, yb)
        perdidas.append(loss)
        correctos += int(acc * len(yb))
        total += len(yb)

    return float(np.mean(perdidas)), correctos / total


def entrenar(args):
    x_train, y_train = cargar_datos(args.data, True, args.train)
    x_test, y_test = cargar_datos(args.data, False, args.test)

    modelo = VisionTransformer(
        in_channels=1,
        image_size=28,
        patch_size=7,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        n_classes=10,
        mlp_hidden_dim=args.mlp,
        seed=args.seed,
    )

    print("Datos train:", x_train.shape, y_train.shape)
    print("Datos test :", x_test.shape, y_test.shape)
    print("Entrenamiento MNIST con ViT")
    print("epoca | loss_train | acc_train")

    for epoca in range(1, args.epocas + 1):
        perdidas = []
        correctos = 0
        total = 0

        for xb, yb in lotes(x_train, y_train, args.batch):
            feat = obtener_features(modelo, xb)
            logits = feat @ modelo.head.weight + modelo.head.bias
            probs = softmax(logits)
            loss, acc = loss_accuracy(logits, yb)

            grad = probs
            grad[np.arange(len(yb)), yb] -= 1.0
            grad /= len(yb)

            grad_w = feat.T @ grad
            grad_b = np.sum(grad, axis=0)

            modelo.head.weight -= args.lr * grad_w
            modelo.head.bias -= args.lr * grad_b

            perdidas.append(loss)
            correctos += int(acc * len(yb))
            total += len(yb)

        print(
            f"{epoca:5d} | "
            f"{np.mean(perdidas):10.4f} | "
            f"{correctos / total:9.4f}"
        )

    loss_test, acc_final = evaluar(modelo, x_test, y_test, args.batch)
    print(f"Resultado final: loss_test={loss_test:.4f} | accuracy={acc_final:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="")
    parser.add_argument("--train", type=int, default=6000)
    parser.add_argument("--test", type=int, default=1000)
    parser.add_argument("--epocas", type=int, default=100)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--mlp", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    entrenar(args)


if __name__ == "__main__":
    main()
