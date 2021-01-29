import numpy as np
import os
import tensorflow as tf
import argparse
import os
from quantized_layers import QuantizedModel

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="mnist")
args = parser.parse_args()

if args.dataset == "fashion-mnist":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
elif args.dataset == "mnist":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
else:
    raise ValueError("Unknown dataset")
y_train = y_train.flatten()
y_test = y_test.flatten()
x_train = x_train.reshape([-1, 28, 28, 1]).astype(np.float32)
x_test = x_test.reshape([-1, 28, 28, 1]).astype(np.float32)

model = QuantizedModel(
    [64, 32],  # 2 hidden layers, fully connected
    input_bits=8,  # Input of (fashion)-MNIST is 8-bit
    quantization_bits=6,  # Network is 6 bit
    last_layer_signed=True,  # Last layer is signed integer
    dropout_rate=0.0,  # Dropout rate (not used)
)

save_path = "weights/{}_mlp.h5".format(args.dataset)

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0005),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.build((None,) + x_train.shape[1:])
model.load_weights(save_path)

_, test_acc = model.evaluate(x=x_test, y=y_test, batch_size=128, verbose=0)
print("Test accuracy is {:0.2f}%".format(100 * test_acc))


y_hat_test = model.predict(x_test)
start_indices = [0, 100, 200, 300]
epsilon = [1, 2, 3, 4]
num_samples = [100, 100, 100, 100] if args.dataset == "mnist" else [100, 100, 50, 50]

# The following loop prints out a list of all test sample indices with corresponding
# attack budget (L-infinity norm) epsilon that need to be checked in this benchmark
for part in range(len(start_indices)):
    for i in range(start_indices[part], start_indices[part] + num_samples[part]):
        if y_test[i] == np.argmax(y_hat_test[i]):
            print(
                "{} test sample {:d} is classified correctly -> Need to check robustness with epsilon {}".format(
                    args.dataset.upper(), i, epsilon[part]
                )
            )
            # ADD YOUR CODE HERE
            # Check robustness of the tf.keras.Model 'model' with respect to the sample 'x_test[i]' and corresponding
            # value of epsilon.
            # The adversarial attack must be untargeted, i.e., changing the prediction away from class 'y_test[i]'
            # to any other class
        else:
            print("{} test sample {:d} MISCLASSIFIED".format(args.dataset.upper(), i))