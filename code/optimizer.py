"""Gradient-ascent optimizers and parameter initializers."""

import numpy as np


class Optimizer:
    """Optimizer base class; call `reset` before use."""
    def __init__(self):
        self.parameters = None

    def reset(self, parameters):
        """Reset this optimizer."""
        self.parameters = parameters

    def step(self, grad, *args, **kwargs):
        """Perform a single optimization step."""
        raise NotImplementedError

class Sga(Optimizer):
    """Stochastic gradient ascent with a constant or scheduled learning rate."""
    def __init__(self, lr):
        super().__init__()
        self.lr = lr
        self.k = 0

    def reset(self, parameters):
        """Reset this optimizer."""
        super().reset(parameters)
        self.k = 0

    def step(self, grad, *args, **kwargs):
        """Perform a single optimization step."""
        lr = self.lr if not callable(self.lr) else self.lr(self.k)
        self.k += 1

        self.parameters += lr * grad


class Initializer:
    """Base class for parameter initializers."""
    def __init__(self):
        pass

    def initialize(self, shape):
        """Create an initial set of parameters."""
        raise NotImplementedError

    def __call__(self, shape):
        """Create an initial set of parameters."""
        return self.initialize(shape)


class Constant(Initializer):
    """Initializes parameters to a constant value."""
    def __init__(self, value=1.0):
        super().__init__()
        self.value = value

    def initialize(self, shape):
        """Create parameters with a fixed initial value."""
        if callable(self.value):
            return np.ones(shape) * self.value(shape)
        else:
            return np.ones(shape) * self.value
