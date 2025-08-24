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
    # Get dimensions
    N, C, H, W = X.shape
    kh, kw = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
    s = stride
    p = padding

    H_p, W_p = H + 2 * p, W + 2 * p
    Xp = np.pad(X, ((0,0), (0,0), (p,p), (p,p)), mode='constant')
    out_h = (H_p - kh) // s + 1
    out_w = (W_p - kw) // s + 1

    cols = np.zeros((C * kh * kw, N * out_h * out_w), dtype=X.dtype)
    col = 0
    for i in range(out_h):
        i_min = i*s
        i_max = i_min + kh
        for j in range(out_w):
            j_min = j*s
            j_max = j_min + kw
            cols[col] = Xp[:, :, i_min:i_max, j_min:j_max].reshape(N, -1).T
            col += N
    return cols, out_h, out_w

def col2im(cols, X_shape, kernel_size, out_h, out_w, stride=1 , padding =1):
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

    H_p, W_p = H + 2 * p, W + 2 * p
    dXp = np.zeros((N, C, H_p, W_p), dtype=cols.dtype)

    col = 0
    for i in range(out_h):
        i_min = i*s
        i_max = i_min + kh
        for j in range(out_w):
            j_min = j*s
            j_max = j_min + kw
            dXp[:, :, i_min:i_max, j_min:j_max] += cols[col:col+N].T.reshape(N, C, kh, kw)
            col += N
    if p == 0:
        return dXp
    return dXp[:, :, p:p+H, p:p+W]


    