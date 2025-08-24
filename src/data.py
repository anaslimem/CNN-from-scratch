import numpy as np
from src.utils import load_cifar10, standardize

class DataLoader:
    def __init__(self, X, y, batch_size=64, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(X))
    
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        for start in range(0, len(self.X), self.batch_size):
            idx = self.indices[start:start + self.batch_size]
            yield self.X[idx], self.y[idx]

def load_cifar10(data_dir="data", batch_size=64, standardize_data=True):
    """
    Load and preprocess the CIFAR-10 dataset.

    Parameters:
        data_dir (str): Directory where CIFAR-10 data is stored or will be downloaded to.
        batch_size (int): Number of samples per batch for the DataLoader.
        standardize_data (bool): Whether to standardize the data to have zero mean and unit variance.

    Returns:
        tuple: (train_loader, test_loader, mean, std)
            - train_loader (DataLoader): DataLoader for training data.
            - test_loader (DataLoader): DataLoader for test data.
            - mean (np.ndarray): Mean used for standardization.
            - std (np.ndarray): Standard deviation used for standardization.
    """
    # Load CIFAR-10 data
    X_train, y_train, X_test, y_test = load_cifar10(data_dir)

    # Standardize data if required
    if standardize_data:
        X_train, mean, std = standardize(X_train)
        X_test, _, _ = standardize(X_test, mean, std)
    # Create DataLoaders
    train_loader = DataLoader(X_train, y_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(X_test, y_test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, mean, std