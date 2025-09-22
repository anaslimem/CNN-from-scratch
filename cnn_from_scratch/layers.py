import numpy as np
from .ops import im2col, col2im, softmax, cross_entropy

class Layer:
    def __init__(self):
        self.params = {}
        self.grads = {}
    def forward(self, x, train=True):
        raise NotImplementedError
    def backward(self, grad_out):
        raise NotImplementedError

# Conv2D (im2col + GEMM)
class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, weight_scale=None):
        super().__init__()
        kh, kw = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kh, kw)
        self.stride = stride
        self.padding = padding

        fan_in = in_channels * kh * kw
        scale = weight_scale if weight_scale is not None else np.sqrt(2.0 / fan_in)
        self.params['W'] = np.random.randn(out_channels, in_channels, kh, kw) * scale
        self.params['b'] = np.zeros((out_channels, 1))

        # caches
        self._cache = {}

    def forward(self, X, training=True):
        # X: (N, C, H, W)
        self.X_shape = X.shape
        N, C, H, W = X.shape
        kh, kw = self.kernel_size

        cols, OH, OW = im2col(X, self.kernel_size, self.stride, self.padding)  # (N*OH*OW, IC*kh*kw)
        # store for backward
        self._cache['cols'] = cols
        self._cache['OH'] = OH
        self._cache['OW'] = OW

        W = self.params['W']  # (OC, IC, kh, kw)
        b = self.params['b']  # (OC, 1)
        W_row = W.reshape(self.out_channels, -1)  # (OC, IC*kh*kw)

        # compute output rows: (N*OH*OW, OC)
        out_rows = cols @ W_row.T  # (N*OH*OW, OC)
        out_rows += b.T  # broadcast bias (1,OC) -> (N*OH*OW, OC)

        # reshape to (N, OC, OH, OW)
        out = out_rows.reshape(N, OH, OW, self.out_channels).transpose(0, 3, 1, 2)
        return out

    def backward(self, grad_out):
        # grad_out: (N, OC, OH, OW)
        N, OC, OH, OW = grad_out.shape
        cols = self._cache['cols']  # (N*OH*OW, IC*kh*kw)
        W = self.params['W']
        W_row = W.reshape(self.out_channels, -1)  # (OC, IC*kh*kw)

        # grad_out rows: (N*OH*OW, OC)
        grad_rows = grad_out.transpose(0,2,3,1).reshape(-1, OC)

        # bias grad
        db = np.sum(grad_rows, axis=0, keepdims=True).T  # (OC,1)
        self.grads['b'] = db

        # dW = grad_rows^T @ cols -> (OC, IC*kh*kw)
        dW_row = grad_rows.T @ cols  # (OC, IC*kh*kw)
        self.grads['W'] = dW_row.reshape(self.params['W'].shape)

        # dcols = grad_rows @ W_row -> (N*OH*OW, IC*kh*kw)
        dcols = grad_rows @ W_row 

        # col2im to get dx shape (N, C, H, W)
        dx = col2im(dcols, self.X_shape, self.kernel_size, self.stride, self.padding)
        return dx

# ReLU
class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self._mask = None

    def forward(self, x, train=True):
        self._mask = (x > 0).astype(x.dtype)
        return x * self._mask

    def backward(self, grad_out):
        return grad_out * self._mask

# MaxPool2D
class MaxPool2D(Layer):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        kh, kw = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.kh, self.kw = kh, kw
        self.stride = stride
        self._cache = {}

    def forward(self, x, train=True):
        # x: (N, C, H, W)
        N, C, H, W = x.shape
        cols, OH, OW = im2col(x, (self.kh, self.kw), self.stride, 0)  # (N*OH*OW, C*kh*kw)

        # reshape -> (N*OH*OW, C, kh*kw)
        cols_resh = cols.reshape(N*OH*OW, C, self.kh*self.kw)

        # take max over last axis
        max_idx = np.argmax(cols_resh, axis=2)  # (N*OH*OW, C)
        out_vals = np.max(cols_resh, axis=2)    # (N*OH*OW, C)

        # store cache
        self._cache['cols_shape'] = cols.shape
        self._cache['max_idx'] = max_idx
        self._cache['OH'] = OH
        self._cache['OW'] = OW
        self._cache['input_shape'] = x.shape

        # reshape output to (N, C, OH, OW)
        out = out_vals.reshape(N, OH, OW, C).transpose(0, 3, 1, 2)
        return out

    def backward(self, grad_out):
        # grad_out: (N, C, OH, OW)
        N, C, OH, OW = grad_out.shape
        cols_shape = self._cache['cols_shape']  # (N*OH*OW, C*kh*kw)
        max_idx = self._cache['max_idx']       # (N*OH*OW, C)

        # grad_out flattened: (N*OH*OW, C)
        grad_flat = grad_out.transpose(0,2,3,1).reshape(-1, C)

        # build grad_cols of shape (N*OH*OW, C, kh*kw)
        grad_cols = np.zeros((N*OH*OW, C, self.kh*self.kw), dtype=grad_out.dtype)
        row_idx = np.arange(N*OH*OW)[:, None]  # (N*OH*OW,1)
        col_idx = np.arange(C)[None, :]        # (1,C)
        grad_cols[row_idx, col_idx, max_idx] = grad_flat

        # reshape to (N*OH*OW, C*kh*kw)
        grad_cols = grad_cols.reshape(N*OH*OW, -1)

        # col2im
        dx = col2im(grad_cols, self._cache['input_shape'], (self.kh, self.kw), self.stride, 0)
        return dx
        
# AvgPool2D
class AvgPool2D(Layer):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        kh, kw = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.kh, self.kw = kh, kw
        self.stride = stride
        self._cache = {}

    def forward(self, x, train=True):
        N, C, H, W = x.shape
        cols, OH, OW = im2col(x, (self.kh, self.kw), self.stride, 0)  # (N*OH*OW, C*kh*kw)
        cols_resh = cols.reshape(N*OH*OW, C, self.kh*self.kw)
        out_vals = np.mean(cols_resh, axis=2)  # (N*OH*OW, C)

        self._cache['OH'] = OH
        self._cache['OW'] = OW
        self._cache['input_shape'] = x.shape

        out = out_vals.reshape(N, OH, OW, C).transpose(0, 3, 1, 2)
        return out

    def backward(self, grad_out):
        N, C, OH, OW = grad_out.shape
        grad_flat = grad_out.transpose(0,2,3,1).reshape(-1, C)  # (N*OH*OW, C)
        # distribute evenly across kh*kw positions
        grad_cols = np.repeat(grad_flat[:, :, None], self.kh*self.kw, axis=2) / (self.kh*self.kw)
        grad_cols = grad_cols.reshape(N*OH*OW, -1)  # (N*OH*OW, C*kh*kw)
        dx = col2im(grad_cols, self._cache['input_shape'], (self.kh, self.kw), self.stride, 0)
        return dx
        
# Flatten
class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.orig_shape = None

    def forward(self, x, train=True):
        self.orig_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad_out):
        return grad_out.reshape(self.orig_shape)

# Linear (Dense)
class Linear(Layer):
    def __init__(self, in_features, out_features, weight_scale=None):
        super().__init__()
        scale = weight_scale if weight_scale is not None else np.sqrt(2.0 / in_features)
        self.params['W'] = np.random.randn(in_features, out_features) * scale
        self.params['b'] = np.zeros((1, out_features))
        self._cache = {}

    def forward(self, x, train=True):
        # x: (N, in_features)
        self._cache['x'] = x
        return x @ self.params['W'] + self.params['b']

    def backward(self, grad_out):
        x = self._cache['x']  # (N, in_features)
        self.grads['W'] = x.T @ grad_out  # (in_features, out_features)
        self.grads['b'] = np.sum(grad_out, axis=0, keepdims=True)
        dx = grad_out @ self.params['W'].T
        return dx

# Dropout (inverted)
class Dropout(Layer):
    def __init__(self, p=0.5):
        super().__init__()
        assert 0.0 <= p < 1.0
        self.p = p
        self.mask = None

    def forward(self, x, train=True):
        if train and self.p > 0:
            self.mask = (np.random.rand(*x.shape) > self.p).astype(x.dtype) / (1.0 - self.p)
            return x * self.mask
        self.mask = None
        return x

    def backward(self, grad_out):
        if self.mask is None:
            return grad_out
        return grad_out * self.mask

# Softmax + CrossEntropy combined
class SoftmaxCrossEntropy(Layer):
    def __init__(self):
        super().__init__()
        self.probs = None
        self.y_true = None

    def forward(self, logits, y_true, training=True):
        # logits: (N, num_classes), y_true: (N,) integer labels
        self.probs = softmax(logits)
        self.y_true = y_true
        return cross_entropy(self.probs, y_true)

    def backward(self, _=None):
        N = self.probs.shape[0]
        grad = self.probs.copy()
        grad[np.arange(N), self.y_true] -= 1.0
        grad /= N
        return grad

# BatchNorm2D (per-channel)
class BatchNorm2D(Layer):
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        super().__init__()
        self.params['gamma'] = np.ones((1, num_features, 1, 1))
        self.params['beta'] = np.zeros((1, num_features, 1, 1))
        self.running_mean = np.zeros((1, num_features, 1, 1))
        self.running_var = np.ones((1, num_features, 1, 1))
        self.momentum = momentum
        self.eps = eps
        self.cache = None

    def forward(self, x, train=True):
        # x: (N, C, H, W)
        if train:
            mean = np.mean(x, axis=(0,2,3), keepdims=True)
            var = np.var(x, axis=(0,2,3), keepdims=True)
            x_hat = (x - mean) / np.sqrt(var + self.eps)
            out = self.params['gamma'] * x_hat + self.params['beta']

            # update running stats
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

            self.cache = (x, x_hat, mean, var)
            return out
        else:
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            return self.params['gamma'] * x_hat + self.params['beta']

    def backward(self, grad_out):
        x, x_hat, mean, var = self.cache
        N, C, H, W = x.shape
        m = N * H * W

        gamma = self.params['gamma']
        # grads
        self.grads['gamma'] = np.sum(grad_out * x_hat, axis=(0,2,3), keepdims=True)
        self.grads['beta'] = np.sum(grad_out, axis=(0,2,3), keepdims=True)

        dxhat = grad_out * gamma
        inv_std = 1.0 / np.sqrt(var + self.eps)

        dx = (1.0 / m) * inv_std * (
            m * dxhat
            - np.sum(dxhat, axis=(0,2,3), keepdims=True)
            - (x - mean) * (inv_std**2) * np.sum(dxhat * (x - mean), axis=(0,2,3), keepdims=True)
        )
        return dx
