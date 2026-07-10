import numpy as np


def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=-1)

    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, n_classes=None):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred)

    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=-1)
    y_pred = y_pred.astype(int)

    if n_classes is None:
        max_label = 0
        if y_true.size:
            max_label = max(max_label, int(np.max(y_true)))
        if y_pred.size:
            max_label = max(max_label, int(np.max(y_pred)))
        n_classes = max_label + 1

    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[true, pred] += 1
    return matrix


def precision_recall_f1(y_true, y_pred, n_classes=None):
    matrix = confusion_matrix(y_true, y_pred, n_classes)
    precision = []
    recall = []
    f1 = []

    for clase in range(matrix.shape[0]):
        tp = matrix[clase, clase]
        fp = np.sum(matrix[:, clase]) - tp
        fn = np.sum(matrix[clase, :]) - tp

        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0

        precision.append(float(p))
        recall.append(float(r))
        f1.append(float(f))

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": float(np.mean(f1)) if f1 else 0.0,
    }
