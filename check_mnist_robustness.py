import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # runs on CPU
import tensorflow as tf
import argparse
from quantized_layers import QuantizedModel
from bv_smt_encoding import QNNEncoding, check_robustness, export_robustness

parser = argparse.ArgumentParser()
parser.add_argument("--sample_id", type=int, default=0)
parser.add_argument("--eps", type=int, default=1)
parser.add_argument("--dataset", default="mnist")
parser.add_argument("--export", action="store_true")
args = parser.parse_args()


if args.dataset == "fashion-mnist":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
elif args.dataset == "mnist":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
else:
    raise ValueError("Unknown dataset '{}'".format(args.dataset))


y_train = y_train.flatten()
y_test = y_test.flatten()

x_train = x_train.reshape([-1, 28* 28]).astype(np.float32)
x_test = x_test.reshape([-1, 28* 28]).astype(np.float32)


model = QuantizedModel(
    [64, 32],
    input_bits=8,
    quantization_bits=6,
    last_layer_signed=True,
)

weight_path = "weights/{}_mlp.h5".format(args.dataset)
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.build((None,28*28)) # force weight allocation
model.load_weights(weight_path)

original_prediction = np.argmax(model.predict(np.expand_dims(x_test[args.sample_id],0))[0])

if original_prediction == y_test[args.sample_id]:
    smt = QNNEncoding(model, verbose=False)
    if args.export:
        os.makedirs("smt2",exist_ok=True)
        filename = "smt2/{}_{:04d}.smt2".format(args.dataset,args.sample_id)
        export_robustness(
            filename,smt, x_test[args.sample_id].flatten(), y_test[args.sample_id], args.eps
          )
        print('Exporting SMTLIB2 file "{}"'.format(filename))
    else:
        is_robust, counterexample = check_robustness(
            smt, x_test[args.sample_id].flatten(), y_test[args.sample_id], args.eps
        )

        if is_robust == True:
            print("{} test sample {} is robust!".format(args.dataset,args.sample_id))
        elif is_robust == False:
            print("{} test sample {} is NOT robust!".format(args.dataset,args.sample_id))
            attacked_prediction = np.argmax(model.predict(np.expand_dims(counterexample,0))[0])
            print("Predicted original class is {}, vs attacked image is predicted as {}".format(original_prediction,attacked_prediction))
        else: # is_robust is None
            print("Could not check {} test sample {}. Timeout!".format(args.dataset,args.sample_id))
else:
    print("{} test sample {} is misclassified!".format(args.dataset,args.sample_id))