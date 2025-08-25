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
        out = out.reshape(self.out_channels, N, self.out_h, self.out_w).transpose(1,0,2,3)
        return out
    
    def backward(self, grad_out):
        N = grad_out.shape[0]
        grad_row = grad_out.transpose(1,0,2,3).reshape(self.out_channels, -1)
        self.grads['b'] = np.sum(grad_row, axis=1, keepdims=True)
        self.grads['W'] = (grad_row @ self.X_cols.T).reshape(self.params['W'].shape)
        W_row = self.params['W'].reshape(self.out_channels, -1)
        dcols = W_row.T @ grad_row
        dx = col2im(dcols, self.X_shape, self.kernel_size, self.out_h, self.out_w, self.stride, self.padding)
        return dx

class ReLU(Layer):
    def forward(self, X, training=True):
        self.mask = (X > 0).astype(X.dtype)
        return X * self.mask
    def backward(self, grad_out):
        return grad_out * self.mask

class MaxPool2D(Layer):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        kh, kw = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.kh = kh
        self.kw = kw
        self.stride = stride
    
    def forward(self, X, training=True):
        self.X_shape = X.shape
        N, C, H, W = X.shape
        cols, out_h, out_w = im2col(X, (self.kh, self.kw), self.stride, 0)
        cols_r = cols.reshape(C*self.kh*self.kw, N*out_h*out_w)
        self.argmax = np.argmax(cols_r, axis=0)
        out = np.max(cols_r, axis=1)
        out = out.reshape(N, C, out_h, out_w).transpose(1, 0, 2, 3)
        self.out_h = out_h
        self.out_w = out_w
        return out
    
    def backward(self, grad_out):
        N, C, out_h, out_w = grad_out.shape
        grad_cols = np.zeros((C*self.kh*self.kw, N*out_h*out_w), dtype=grad_out.dtype)
        flat = grad_out.transpose(1,0,2,3).reshape(C, -1)
        idx = self.argmax
        rows = np.arange(C)[:, None]
        cols_index = np.arange(N*out_h*out_w)[None, :]
        grad_cols[rows, idx, cols_index] = flat
        grad_cols = grad_cols.reshape(C, self.kh, self.kw, N, out_h, out_w)
        dx = col2im(grad_cols, self.X_shape, (self.kh, self.kw), self.out_h, self.out_w, self.stride, 0)
        return dx
    
class AvgPool2D(Layer):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        kh, kw = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.kh = kh
        self.kw = kw
        self.stride = stride

    def forward(self, X, training=True):
        self.X_shape = X.shape
        N, C, H, W = X.shape
        cols, out_h, out_w = im2col(X, (self.kh, self.kw), self.stride, 0)
        cols_r = cols.reshape(C*self.kh*self.kw, N*out_h*out_w)
        out = np.mean(cols_r, axis=0)
        out = out.reshape(N, C, out_h, out_w).transpose(1, 0, 2, 3)
        self.out_h = out_h
        self.out_w = out_w
        return out
    
    def backward(self, grad_out):
        N, C, out_h, out_w = grad_out.shape
        grad_cols = np.repeat(grad_out.transpose(1,0,2,3).reshape(C, 1, -1), self.kh*self.kw, axis=1)
        grad_cols /= (self.kh * self.kw)
        grad_cols = grad_cols.reshape(C*self.kh*self.kw, N*out_h*out_w)
        dx = col2im(grad_cols, self.X_shape, (self.kh, self.kw), self.out_h, self.out_w, self.stride, 0)
        return dx
    
class Dropout(Layer):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.mask = None

    def forward(self, X, training=True):
        if training and self.p > 0:
            self.mask = (np.random.rand(*X.shape) > self.p).astype(X.dtype) / (1.0 - self.p)
            return X * self.mask
        self.mask = None
        return X
        
    def backward(self, grad_out):
        if self.mask is None:
            return grad_out
        return grad_out * self.mask
    
class Flatten(Layer):
    def forward(self, X, training=True):
        self.orig = X.shape
        return X.reshape(X.shape[0], -1)
    
    def backward(self, grad_out):
        return grad_out.reshape(self.orig)

class Linear(Layer):
    def __init__(self, in_features, out_features, weight_scale=None):
        super().__init__()
        scale = weight_scale if weight_scale is not None else np.sqrt(2. / in_features)
        self.params['W'] = np.random.randn(out_features, in_features) * scale
        self.params['b'] = np.zeros((1, out_features))
    
    def forward(self, X, training=True):
        self.X = X
        return X @ self.params['W'] + self.params['b']

    def backward(self, grad_out):
        self.grads['W'] = self.X.T @ grad_out
        self.grads['b'] = np.sum(grad_out, axis=0, keepdims=True)
        return grad_out @ self.params['W'].T
    

    
