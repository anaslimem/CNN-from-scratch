
import numpy as np
import tensorflow as tf

def load_mnist_data_from_tensorflow():
    """
    Load the MNIST dataset using TensorFlow and return a data loader.
    """
    # The data is already split into training and testing sets.
    # This function returns data in the format (samples, height, width).
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # The original image data is of type uint8 with pixel values from 0-255.
    # We normalize the data by converting it to float32 and scaling it by 255.0.
    # This ensures pixel values are in the range [0.0, 1.0].
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    # The most common issue is that the data is not in the correct shape for
    # a convolutional layer. We must add a channel dimension.
    # The shape will change from (samples, 28, 28) to (samples, 28, 28, 1).
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    
    # We will now transpose the data to match the expected format of (N, C, H, W)
    # for the im2col function. This is a crucial step.
    x_train = np.transpose(x_train, (0, 3, 1, 2))
    x_test = np.transpose(x_test, (0, 3, 1, 2))

    return (x_train, y_train), (x_test, y_test)

