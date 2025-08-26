
import numpy as np
from .layers import Conv2D, ReLU, MaxPool2D, Flatten, Linear, Dropout, SoftmaxCrossEntropy, BatchNorm2D
from .model import Sequential
from .optim import SGD, AdamW
from .utils import set_seed
from .data import load_cifar10_loaders

def build_small_cnn(use_batchnorm=False, dropout_p=0.5):
    layers = [
        Conv2D(3, 32, kernel_size=3, stride=1, padding=1),
        (BatchNorm2D(32) if use_batchnorm else None),
        ReLU(),
        MaxPool2D(kernel_size=2, stride=2),
        Conv2D(32, 64, kernel_size=3, stride=1, padding=1),
        (BatchNorm2D(64) if use_batchnorm else None),
        ReLU(),
        MaxPool2D(kernel_size=2, stride=2),
        Flatten(),
        Dropout(p=dropout_p),
        Linear(64*8*8, 10)
    ]
    return Sequential([l for l in layers if l is not None])

def train(num_epochs=10, batch_size=128, lr=0.01, weight_decay=5e-4, dropout_p=0.5,
          optimizer="sgd", use_batchnorm=False, seed=42, decoupled=False, data_path="data",
          save_plots=True):
    set_seed(seed)
    train_loader, test_loader = load_cifar10_loaders(path=data_path, batch_size=batch_size, standardize=True)
    model = build_small_cnn(use_batchnorm=use_batchnorm, dropout_p=dropout_p)
    loss_fn = SoftmaxCrossEntropy()

    if optimizer == "sgd":
        opt = SGD(lr=lr, momentum=0.9, weight_decay=weight_decay, decoupled=decoupled)
    else:
        opt = AdamW(lr=lr, weight_decay=weight_decay)

    train_losses, test_accs = [], []

    for epoch in range(1, num_epochs+1):
        running_loss = 0.0
        n = 0
        for X, y in train_loader:
            logits = model.forward(X, train=True)
            loss = loss_fn.forward(logits, y, train=True)
            grad = loss_fn.backward()
            model.backward(grad)
            opt.step(model.params_and_grads)
            running_loss += loss * X.shape[0]
            n += X.shape[0]
        avg_loss = running_loss / n
        train_losses.append(avg_loss)

        # Eval
        correct, total = 0, 0
        for X, y in test_loader:
            logits = model.forward(X, train=False)
            pred = np.argmax(logits, axis=1)
            correct += (pred == y).sum()
            total += y.size
        acc = correct / total
        test_accs.append(acc)
        print(f"Epoch {epoch}/{num_epochs} - loss: {avg_loss:.4f} - test acc: {acc*100:.2f}%")

    if save_plots:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(range(1, num_epochs+1), train_losses, label="train loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
        plt.savefig("plots/train_loss.png"); plt.close()

        plt.figure()
        plt.plot(range(1, num_epochs+1), test_accs, label="test acc")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.tight_layout()
        plt.savefig("plots/test_acc.png"); plt.close()

    return train_losses, test_accs
