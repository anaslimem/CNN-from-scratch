from cnn_from_scratch.train import train
if __name__ == "__main__":
    # You can adjust these parameters as needed
    num_epochs = 10
    batch_size = 128
    lr = 0.01
    weight_decay = 5e-4
    dropout_p = 0.5
    optimizer = "sgd"  # or "adamw"
    use_batchnorm = True
    seed = 42
    decoupled = False
    save_plots = True

    train_losses, test_accs = train(
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        dropout_p=dropout_p,
        optimizer=optimizer,
        use_batchnorm=use_batchnorm,
        seed=seed,
        decoupled=decoupled,
        save_plots=save_plots
    )

    print("Training complete.")
    print("Final test accuracy: {:.2f}%".format(test_accs[-1] * 100))