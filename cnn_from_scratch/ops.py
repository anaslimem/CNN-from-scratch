
import numpy as np

def im2col(X, kernel_size, stride, padding):
    # Expect input X in channels-first format: (N, C, H, W)
    N, C, H, W = X.shape
    KH, KW = kernel_size
    # Calculate output height/width
    OH = (H + 2 * padding - KH) // stride + 1
    OW = (W + 2 * padding - KW) // stride + 1

    # Pad input on height and width dims
    X_padded = np.pad(X, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")

    # cols will hold patches with shape (N, C, KH, KW, OH, OW)
    cols = np.zeros((N, C, KH, KW, OH, OW), dtype=X.dtype)
    for y in range(KH):
        y_max = y + stride * OH
        for x in range(KW):
            x_max = x + stride * OW
            cols[:, :, y, x, :, :] = X_padded[:, :, y:y_max:stride, x:x_max:stride]

    # Transpose to (N, OH, OW, C, KH, KW) then reshape to (N*OH*OW, C*KH*KW)
    cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * OH * OW, -1)
    return cols, OH, OW

def col2im(cols, X_shape, kernel_size, stride, padding):
    N, C, H, W = X_shape
    KH, KW = kernel_size
    OH = (H + 2*padding - KH) // stride + 1
    OW = (W + 2*padding - KW) // stride + 1

    cols_reshaped = cols.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)
    X_padded = np.zeros((N, C, H + 2*padding, W + 2*padding))

    for y in range(KH):
        y_max = y + stride*OH
        for x in range(KW):
            x_max = x + stride*OW
            X_padded[:, :, y:y_max:stride, x:x_max:stride] += cols_reshaped[:, :, y, x, :, :]
    return X_padded[:, :, padding:H+padding, padding:W+padding]

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)

def cross_entropy(probs, y_true):
    N = probs.shape[0]
    correct_logprobs = -np.log(probs[range(N), y_true] + 1e-9)
    return np.sum(correct_logprobs) / N
