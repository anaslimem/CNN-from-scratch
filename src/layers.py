import numpy as np
from src.ops import im2col, col2im, softmax, cross_entropy

class Layer:
    def __init__(self):
        self.params = {}
        self.grads = {}
    def forward(self, X, training=True):
        raise NotImplementedError

    def backward(self, dOut):
        raise NotImplementedError

class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, weight_scale=None):
        super().__init__()
        kh, kw = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        fan_in = in_channels * kh * kw
        scale = weight_scale if weight_scale is not None else np.sqrt(2. / fan_in)
        self.params['W'] = np.random.randn(out_channels, in_channels, kh, kw) * scale
        self.params['b'] = np.zeros((out_channels, 1))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kh, kw)
        self.stride = stride
        self.padding = padding

    def forward(self, X, training=True):
        self.X_shape = X.shape
        self.X_cols, self.out_h, self.out_w = im2col(X, self.kernel_size, self.stride, self.padding)
        W = self.params['W']
        b = self.params['b']
        W_row = W.reshape(self.out_channels, -1)
        out = (W_row @ self.X_cols + b)
        N = X.shape[0]
        return out.reshape(self.out_channels, N, self.out_h, self.out_w).transpose(1,0,2,3)
        return out
