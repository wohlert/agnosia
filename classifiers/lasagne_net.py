import configparser

import numpy as np
np.random.seed(1234)

import theano
import theano.tensor as T

import lasagne
from lasagne.layers import Conv2DLayer, MaxPool2DLayer, InputLayer
from lasagne.layers import DenseLayer, FlattenLayer
from lasagne.layers import ConcatLayer, ReshapeLayer, get_output_shape
from lasagne.layers import Conv1DLayer, LSTMLayer, SliceLayer, DimshuffleLayer

config = configparser.ConfigParser()
config.read("settings.ini")

IMAGE_SIZE = int(config['default']["resolution"])
FRAMES = int(config['default']["frames"])
image_dir = str(config['default']["output"])

def build_cnn(input_var, W_init=None, n_layers=(4, 2, 1), n_filters=32):
    """
    Builds a VGG style CNN network followed by a fully-connected layer and a softmax layer.
    Stacks are separated by a maxpool layer. Number of kernels in each layer is twice
    the number in previous stack.
    """

    weights = []
    count = 0

    # If no initial weight is given, initialize with GlorotUniform
    if W_init is None:
        W_init = [lasagne.init.GlorotUniform()] * sum(n_layers)

    # Input layer
    # Tensor(?, channels, width, height)
    network = InputLayer(shape=(None, 3, IMAGE_SIZE, IMAGE_SIZE), input_var=input_var)

    # 4 Conv(32*1 =  32) -> Max
    # 2 Conv(32*2 =  64) -> Max
    # 1 Conv(32*4 = 128) -> Max
    for i, s in enumerate(n_layers):
        for l in range(s):
            network = Conv2DLayer(network, num_filters=n_filters * (2 ** i), filter_size=(3, 3),
                          W=W_init[count], pad='same')
            count += 1
            weights.append(network.W)
        network = MaxPool2DLayer(network, pool_size=(2, 2))

    return network, weights


def build_convpool_mix(input_vars, grad_clip=100):
    """
    Builds the complete network with LSTM and 1D-conv layers combined
    """
    convnets = []
    W_init = None
    n_frames = FRAMES

    # Build 7 parallel CNNs with shared weights
    for i in range(n_frames):
        if i == 0:
            convnet, W_init = build_cnn(input_vars[i])
        else:
            convnet, _ = build_cnn(input_vars[i], W_init)

        convnets.append(FlattenLayer(convnet))

    # at this point convnets shape is [numTimeWin][n_samples, features]
    # we want the shape to be [n_samples, features, numTimeWin]
    convpool = ConcatLayer(convnets)
    # convpool = ReshapeLayer(convpool, ([0], -1, numTimeWin))

    convpool = ReshapeLayer(convpool, ([0], n_frames, get_output_shape(convnets[0])[1]))
    reformConvpool = DimshuffleLayer(convpool, (0, 2, 1))

    # input to 1D convlayer should be in (batch_size, num_input_channels, input_length)
    conv_out = Conv1DLayer(reformConvpool, 64, 3)
    conv_out = FlattenLayer(conv_out)
    # Input to LSTM should have the shape as (batch size, SEQ_LENGTH, num_features)
    lstm = LSTMLayer(convpool, num_units=128, grad_clipping=grad_clip,
        nonlinearity=lasagne.nonlinearities.tanh)
    # After LSTM layer you either need to reshape or slice it (depending on whether you
    # want to keep all predictions or just the last prediction.
    # http://lasagne.readthedocs.org/en/latest/modules/layers/recurrent.html
    # https://github.com/Lasagne/Recipes/blob/master/examples/lstm_text_generation.py
    # lstm_out = SliceLayer(convpool, -1, 1)        # bypassing LSTM
    lstm_out = SliceLayer(lstm, -1, 1)

    # Merge 1D-Conv and LSTM outputs
    dense_input = ConcatLayer([conv_out, lstm_out])
    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    convpool = DenseLayer(lasagne.layers.dropout(dense_input, p=.5),
                          num_units=512, nonlinearity=lasagne.nonlinearities.rectify)
    # We only need the final prediction, we isolate that quantity and feed it
    # to the next layer.

    # And, finally, the 1-unit output layer with 50% dropout on its inputs:
    convpool = DenseLayer(convpool,
                          num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)

    return convpool

def load_data(base):
    import pandas as pd
    from scipy.misc import imread
    from sklearn.cross_validation import train_test_split

    df = pd.read_csv("labels.csv")[:100]
    y = df['label'].values
    names = df['name'].values.astype(str)

    # Load "images/train_subject1/trial1.0.jpeg"
    X = np.array([imread("{}{}.{}.jpeg".format(base, i, f)) for i in names for f in range(FRAMES)])
    X = X.astype('float32')
    X /= 255

    X = X.reshape(-1, FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE)

    return train_test_split(X, y)


X_train, X_test, y_train, y_test = load_data(image_dir)

input_var = T.TensorType('floatX', (False,) * 5)()
target_var = T.ivector('targets')

model = build_convpool_mix(input_var, grad_clip=90)

prediction = lasagne.layers.get_output(model)
loss = lasagne.objectives.binary_crossentropy(prediction, target_var)
loss = loss.mean() + 1e-4 * lasagne.regularization.regularize_network_params(model, lasagne.regularization.l2)

params = lasagne.layers.get_all_params(model, trainable=True)
updates = lasagne.updates.adam(loss, params)

train_fn = theano.function([input_var, target_var], loss, updates=updates, allow_input_downcast=True)

nb_epoch = 15
batch_size = 32

train_length = len(y_train)

for epoch in range(nb_epoch):
    loss = 0
    total_batch = int(train_length/batch_size)
    for i in range(total_batch):
        current_batch = range(i*batch_size, ((i+1)*batch_size))
        input_batch = X_train[current_batch].reshape(-1, FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE)
        target_batch = y_train[current_batch].ravel()

        print("Input: ", input_batch.shape)
        print("Target: ", target_batch.shape)

        loss += train_fn(input_batch, target_batch)

    print("Epoch: {} loss: {}".format(epoch+1, loss/train_length))

test_prediction = lasagne.layers.get_output(model, deterministic=True)
test_fn = theano.function([input_var], T.argmax(test_prediciton, axis=1))

from sklearn.metrics import accuracy_score
print(accuracy_score(predict_fn(X_test), y_test.ravel()))
