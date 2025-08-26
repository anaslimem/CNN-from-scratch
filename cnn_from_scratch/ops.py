
import numpy as np

def im2col(X, kernel_size, stride=1, padding=0):
    """
    Convert 4D input X (N, C, H, W) into columns for GEMM-based convolution.
    Args:
        X: ndarray of shape (N, C, H, W)
        kernel_size: (kh, kw)
        stride: int
        padding: int
    Returns:
        cols: (C*kh*kw, N*out_h*out_w)
        out_h, out_w: output spatial dims
    """
    N, C, H, W = X.shape
    kh, kw = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
    s = stride
    p = padding

    H_p, W_p = H + 2*p, W + 2*p
    Xp = np.pad(X, ((0,0),(0,0),(p,p),(p,p)), mode="constant")
    out_h = (H_p - kh)//s + 1
    out_w = (W_p - kw)//s + 1

    cols = np.zeros((C*kh*kw, N*out_h*out_w), dtype=Xp.dtype)
    col = 0
    for i in range(out_h):
        i_min = i*s
        i_max = i*s + kh
        for j in range(out_w):
            j_min = j*s
            j_max = j*s + kw
            patch = Xp[:, :, i_min:i_max, j_min:j_max]  # (N,C,kh,kw)
            cols[:, col:col+N] = patch.reshape(N, -1).T
            col += N
    return cols, out_h, out_w

def col2im(cols, X_shape, kernel_size, out_h, out_w, stride=1, padding=0):
    """
    Inverse of im2col for accumulating gradients.
    Args:
        cols: (C*kh*kw, N*out_h*out_w)
        X_shape: (N, C, H, W)
    Returns:
        dX: (N, C, H, W)
    """
    N, C, H, W = X_shape
    kh, kw = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
    s = stride
    p = padding

    H_p, W_p = H + 2*p, W + 2*p
    Xp = np.zeros((N, C, H_p, W_p), dtype=cols.dtype)

    col = 0
    for i in range(out_h):
        i_min = i*s
        i_max = i*s + kh
        for j in range(out_w):
            j_min = j*s
            j_max = j*s + kw
            patch = cols[:, col:col+N].T.reshape(N, C, kh, kw)
            Xp[:, :, i_min:i_max, j_min:j_max] += patch
            col += N

    if p == 0:
        return Xp
    return Xp[:, :, p:p+H, p:p+W]

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)

def cross_entropy(probs, y_true):
    N = probs.shape[0]
    if y_true.ndim == 1:
        p = np.clip(probs[np.arange(N), y_true], 1e-12, 1.0)
        return -np.mean(np.log(p))
    else:
        p = np.clip(probs, 1e-12, 1.0)
        return -np.mean(np.sum(y_true * np.log(p), axis=1))
