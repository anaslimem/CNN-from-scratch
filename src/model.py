
class Sequential:
    def __init__(self, layers):
        self.layers = layers
    
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

    def backward(self, grad_out):
        for layer in reversed(self.layers):
            grad_out = layer.backward(grad_out)
        return grad_out

    def zero_grad(self):
        for layer in self.layers:
            for k in getattr(layer, 'grads', {}):
                layer.grads[k] = None
