
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

    def forward(self, x, train=True):
        for layer in self.layers:
            x = layer.forward(x, train=train)
        return x

    def backward(self, grad_out):
        for layer in reversed(self.layers):
            grad_out = layer.backward(grad_out)
        return grad_out

    def zero_grads(self):
        for layer in self.layers:
            for k in getattr(layer, 'grads', {}):
                layer.grads[k] = None
