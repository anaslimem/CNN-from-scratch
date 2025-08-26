import numpy as np

class Sequential:
    def __init__(self, layers, optimizer=None):
        self.layers = layers
        self.optimizer = optimizer   # optimizer attached to the model
    
    @property
    def params_and_grads(self):
        for layer in self.layers:
            for name, param in getattr(layer, 'params', {}).items():
                grad = layer.grads.get(name, None)
                if grad is not None:
                    yield param, grad
    
    def forward(self, X, training=True):
        for layer in self.layers:
            X = layer.forward(X, training)
        return X
    
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def step(self):
        if self.optimizer is None:
            raise ValueError("No optimizer attached to model")
        self.optimizer.step(self.params_and_grads)
    
    def zero_grad(self):
        for layer in self.layers:
            if hasattr(layer, 'grads'):
                for name in layer.grads:
                    layer.grads[name] = None
