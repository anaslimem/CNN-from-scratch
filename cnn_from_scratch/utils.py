
import numpy as np
import os, pickle, tarfile, urllib.request

def set_seed(seed=42):
    np.random.seed(seed)

def one_hot(y, num_classes):
    oh = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    oh[np.arange(y.shape[0]), y] = 1.0
    return oh

def accuracy(logits, y):
    preds = np.argmax(logits, axis=1)
    return np.mean(preds == y)

def standardize_per_channel(X, mean=None, std=None):
    # X: (N, C, H, W)
    if mean is None:
        mean = X.mean(axis=(0,2,3), keepdims=True)
    if std is None:
        std = X.std(axis=(0,2,3), keepdims=True) + 1e-8
    return (X - mean) / std, mean, std

def download_cifar10(dest="data"):
    os.makedirs(dest, exist_ok=True)
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    out = os.path.join(dest, "cifar-10-python.tar.gz")
    if not os.path.exists(out):
        urllib.request.urlretrieve(url, out)
    extracted = os.path.join(dest, "cifar-10-batches-py")
    if not os.path.exists(extracted):
        with tarfile.open(out, "r:gz") as tar:
            tar.extractall(path=dest)
    return extracted

def load_cifar10(path="data", split="train"):
    root = download_cifar10(dest=path)
    def _load_batch(fp):
        with open(fp, "rb") as f:
            d = pickle.load(f, encoding="bytes")
            X = d[b"data"]
            y = np.array(d[b"labels"], dtype=np.int64)
            X = X.reshape(-1,3,32,32).astype(np.float32)/255.0
            return X, y
    if split == "train":
        imgs, labels = [], []
        for i in range(1,6):
            Xb, yb = _load_batch(os.path.join(root, f"data_batch_{i}"))
            imgs.append(Xb); labels.append(yb)
        X = np.concatenate(imgs, axis=0)
        y = np.concatenate(labels, axis=0)
        return X, y
    elif split == "test":
        return _load_batch(os.path.join(root, "test_batch"))
    else:
        raise ValueError("split must be 'train' or 'test'")
