import numpy as np
import os, pickle, tarfile, urllib.request

def set_seed(seed=42):
    """
    Set the random seed for NumPy's random number generator.

    Parameters:
        seed (int): The seed value to use for reproducibility. Default is 42.
    """
    np.random.seed(seed)

def one_hot(y, num_classes):
    """
    Convert integer class labels to one-hot encoded vectors.

    Parameters:
        y (np.ndarray): Array of integer class labels of shape (N,).
        num_classes (int): Total number of classes.

    Returns:
        np.ndarray: One-hot encoded array of shape (N, num_classes).
    """
    oh = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    oh[np.arange(y.shape[0]), y] = 1.0
    return oh

def accuracy(logits, y):
    """
    Compute the accuracy of predictions.

    Parameters:
        logits (np.ndarray): Array of predicted logits of shape (N, num_classes).
        y (np.ndarray): Array of true class labels of shape (N,).

    Returns:
        float: The accuracy as a proportion of correct predictions.
    """
    preds = np.argmax(logits, axis=1)
    return np.mean(preds == y)

def standardize(X, mean=None, std=None):
    """
    Standardize the input data to have zero mean and unit variance.

    Parameters:
        X (np.ndarray): Input data of shape (N, D).
        mean (np.ndarray or None): Mean values for each feature. If None, compute from X.
        std (np.ndarray or None): Standard deviation values for each feature. If None, compute from X.

    Returns:
        np.ndarray: Standardized data of shape (N, D).
        np.ndarray: Mean values used for standardization.
        np.ndarray: Standard deviation values used for standardization.
    """
    if mean is None:
        mean = X.mean(axis=(0,2,3), keepdims=True)
    if std is None:
        std = X.std(axis=(0,2,3), keepdims=True)
    return (X - mean) /  std, mean, std

def download_cifar10(data_dir="data"):
    """
    Download and extract the CIFAR-10 dataset.

    Parameters:
        data_dir (str): Directory to store the downloaded data.
    """
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tar_path = os.path.join(data_dir, "cifar-10-python.tar.gz")

    # Download the dataset if it doesn't exist
    if not os.path.exists(tar_path):
        urllib.request.urlretrieve(url, tar_path)
    extracted = os.path.join(data_dir, "cifar-10-batches-py")
    if not os.path.exists(extracted):
    # Extract the dataset
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=data_dir)
    print("CIFAR-10 dataset downloaded and extracted.")
    return extracted

def load_cifar10(path="data", split="train"):
    """
    Load the CIFAR-10 dataset.

    Parameters:
        path (str): Directory where the CIFAR-10 data is stored.
        split (str): Which split to load ('train' or 'test').

    Returns:
        X (np.ndarray): Array of images of shape (N, 3, 32, 32).
        y (np.ndarray): Array of labels of shape (N,).
    """
    root = download_cifar10(data_dir=path)
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
