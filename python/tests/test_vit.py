import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from layers.mlp import MLP
from model.encoder_block import TransformerEncoderBlock
from training.metrics import accuracy_score, confusion_matrix, precision_recall_f1
from training.trainer import Trainer
from utils.weight_init import init_weights_vit


def test_mlp_preserves_token_shape():
    x = np.random.randn(2, 5, 8)
    mlp = MLP(d_model=8, hidden_dim=16, seed=7)

    out = mlp(x)

    assert out.shape == x.shape
    assert np.issubdtype(out.dtype, np.floating)


def test_encoder_block_preserves_shape_and_returns_attention_weights():
    x = np.random.randn(2, 4, 8)
    block = TransformerEncoderBlock(d_model=8, n_heads=2, mlp_hidden_dim=16, seed=11)

    out, weights = block(x, devolver_pesos=True)

    assert out.shape == x.shape
    assert len(weights) == 2
    assert len(weights[0]) == 2
    assert len(weights[0][0]) == 4
    assert len(weights[0][0][0]) == 4


def test_encoder_block_accepts_python_lists():
    x = np.random.randn(1, 3, 4).tolist()
    block = TransformerEncoderBlock(d_model=4, n_heads=2, mlp_hidden_dim=8, seed=3)

    out = block(x)

    assert isinstance(out, list)
    assert len(out) == 1
    assert len(out[0]) == 3
    assert len(out[0][0]) == 4


def test_metrics_for_classification_outputs():
    y_true = np.array([0, 1, 1, 2])
    logits = np.array([
        [0.9, 0.1, 0.0],
        [0.1, 0.7, 0.2],
        [0.2, 0.6, 0.2],
        [0.1, 0.2, 0.7],
    ])

    assert accuracy_score(y_true, logits) == 1.0
    assert confusion_matrix(y_true, logits).shape == (3, 3)
    assert precision_recall_f1(y_true, logits)["macro_f1"] == 1.0


def test_trainer_runs_epoch_with_numpy_model():
    class DummyModel:
        def __call__(self, x, training=False):
            return x

    def mse(pred, target):
        return np.mean((np.asarray(pred) - np.asarray(target)) ** 2)

    data = [
        (np.array([[0.8, 0.2], [0.1, 0.9]]), np.array([0, 1])),
    ]
    trainer = Trainer(DummyModel(), mse)

    result = trainer.train_one_epoch(data)

    assert "loss" in result
    assert "metric" in result
    assert len(trainer.history["train_loss"]) == 1


def test_init_weights_vit_updates_numpy_arrays():
    mlp = MLP(d_model=4, hidden_dim=8, seed=1)
    old_w1 = mlp.w1.copy()

    init_weights_vit(mlp, seed=2)

    assert mlp.w1.shape == old_w1.shape
    assert not np.array_equal(mlp.w1, old_w1)
    assert np.allclose(mlp.b1, 0.0)
