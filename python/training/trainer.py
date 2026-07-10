import pickle

import numpy as np

from training.metrics import accuracy_score


class Trainer:
    def __init__(self, model, loss_fn, optimizer=None, metric_fn=accuracy_score):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metric_fn = metric_fn
        self.history = {"train_loss": [], "train_metric": [], "val_loss": [], "val_metric": []}

    def _forward(self, x, training=False):
        try:
            return self.model(x, training=training)
        except TypeError:
            return self.model(x)

    def _run_epoch(self, dataloader, training=False):
        losses = []
        predictions = []
        targets = []

        for x, y in dataloader:
            if training and self.optimizer is not None and hasattr(self.optimizer, "zero_grad"):
                self.optimizer.zero_grad()

            y_pred = self._forward(x, training=training)
            loss = self.loss_fn(y_pred, y)
            losses.append(float(loss))

            if training and self.optimizer is not None:
                if hasattr(loss, "backward"):
                    loss.backward()
                if hasattr(self.optimizer, "step"):
                    self.optimizer.step()
                elif callable(self.optimizer):
                    self.optimizer(self.model, loss)

            predictions.extend(np.asarray(y_pred).tolist())
            targets.extend(np.asarray(y).tolist())

        metric = self.metric_fn(np.asarray(targets), np.asarray(predictions)) if targets else 0.0
        mean_loss = float(np.mean(losses)) if losses else 0.0
        return mean_loss, metric

    def train_one_epoch(self, train_loader):
        loss, metric = self._run_epoch(train_loader, training=True)
        self.history["train_loss"].append(loss)
        self.history["train_metric"].append(metric)
        return {"loss": loss, "metric": metric}

    def validate(self, val_loader):
        loss, metric = self._run_epoch(val_loader, training=False)
        self.history["val_loss"].append(loss)
        self.history["val_metric"].append(metric)
        return {"loss": loss, "metric": metric}

    def fit(self, train_loader, val_loader=None, epochs=1, patience=None):
        best_val = None
        epochs_without_improvement = 0

        for _ in range(epochs):
            train_result = self.train_one_epoch(train_loader)
            val_result = self.validate(val_loader) if val_loader is not None else None

            if val_result is not None and patience is not None:
                current = val_result["loss"]
                if best_val is None or current < best_val:
                    best_val = current
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        break

        return self.history

    def save_checkpoint(self, path):
        with open(path, "wb") as file:
            pickle.dump({"model": self.model, "history": self.history}, file)

    def load_checkpoint(self, path):
        with open(path, "rb") as file:
            checkpoint = pickle.load(file)
        self.model = checkpoint["model"]
        self.history = checkpoint.get("history", self.history)
        return checkpoint
