import numpy as np

def accuracy(y_pred, y_true):
    """
    Compute accuracy between predictions and true labels.
    y_pred: (N, num_classes) logits or probabilities
    y_true: (N,) ground-truth labels
    """
    preds = np.argmax(y_pred, axis=1)
    return np.mean(preds == y_true)


def one_hot(y, num_classes):
    """
    Convert labels (N,) into one-hot encoded matrix (N, num_classes).
    """
    one_hot_encoded = np.zeros((y.shape[0], num_classes))
    one_hot_encoded[np.arange(y.shape[0]), y] = 1
    return one_hot_encoded


def batch_iterator(X, y, batch_size, shuffle=True):
    """
    Generate mini-batches of (X_batch, y_batch).
    """
    N = X.shape[0]
    indices = np.arange(N)
    if shuffle:
        np.random.shuffle(indices)
    
    for start in range(0, N, batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]

def set_seed(seed):
    np.random.seed(seed)

    