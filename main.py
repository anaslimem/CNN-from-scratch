
import argparse
from cnn_from_scratch.train import train

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--optimizer", choices=["sgd","adamw"], default="sgd")
    p.add_argument("--batchnorm", action="store_true")
    p.add_argument("--decoupled", action="store_true", help="use decoupled decay with SGD")
    p.add_argument("--data_path", type=str, default="data")
    args = p.parse_args()

    train(num_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
          weight_decay=args.weight_decay, dropout_p=args.dropout, optimizer=args.optimizer,
          use_batchnorm=args.batchnorm, decoupled=args.decoupled, data_path=args.data_path)