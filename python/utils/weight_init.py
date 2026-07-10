import math

import numpy as np


def truncated_normal(shape, mean=0.0, std=0.02, limit=2.0, seed=None):
    rng = np.random.default_rng(seed)
    values = rng.normal(mean, std, shape)
    lower = mean - limit * std
    upper = mean + limit * std
    return np.clip(values, lower, upper)


def xavier_uniform(shape, seed=None):
    if len(shape) < 2:
        return np.zeros(shape)

    fan_in, fan_out = shape[0], shape[1]
    bound = math.sqrt(6.0 / (fan_in + fan_out))
    rng = np.random.default_rng(seed)
    return rng.uniform(-bound, bound, shape)


def init_weights_vit(model, seed=None):
    rng = np.random.default_rng(seed)

    for value in vars(model).values():
        if hasattr(value, "shape") and isinstance(value, np.ndarray):
            if value.ndim >= 2:
                value[...] = xavier_uniform(value.shape, seed=int(rng.integers(0, 1_000_000)))
            else:
                value[...] = 0.0
        elif hasattr(value, "__dict__"):
            init_weights_vit(value, seed=int(rng.integers(0, 1_000_000)))

    return model
