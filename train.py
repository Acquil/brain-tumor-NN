"""
Utility used by the Network class to actually train.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

"""
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout
from PIL import Image
import numpy as np
import cv2
import csv
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

# Helper: Early stopping.
early_stopper = EarlyStopping(patience=5)

def get_cifar10():
    """Retrieve the CIFAR dataset and process the data."""
    # Set defaults.
    nb_classes = 10
    batch_size = 64
    input_shape = (3072,)

    # Get the data.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.reshape(50000, 3072)
    x_test = x_test.reshape(10000, 3072)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)

def get_mnist():
    """Retrieve the MNIST dataset and process the data."""
    # Set defaults.
    nb_classes = 10
    batch_size = 128
    input_shape = (784,)

    # Get the data.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)


def get_data_small():
    nb_classes = 2
    batch_size = 5
    input_shape = (128*128,)

    def fetch_image(item):
        image = Image.open(item)
        # convert image to monochrome
        image = image.convert('L')
        # convert image to numpy array
        data = np.asarray(image)
        # print(data.shape)
        return data

    def replace_image(x):
        # x = x.astype('float32')
        return np.fromiter((fetch_image(xi) for xi in x), x.dtype)

    x_train = []
    y_train = []

    x_test = []
    y_test = []

    with open('meta/train.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            x_train.append(row[0])
            y_train.append(row[1])

    with open('meta/test.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            x_test.append(row[0])
            y_test.append(row[1])

    train_length = len(y_train)
    test_length = len(y_test)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)


    x_train = list(map(lambda x:fetch_image(x),x_train))
    x_train = np.asarray(x_train,dtype=np.float32)
    x_test = list(map(lambda x:fetch_image(x),x_test))
    x_test = np.asarray(x_test,dtype=np.float32)

    reshape_pixels = 128*128
    x_train = x_train.reshape(train_length, reshape_pixels)
    x_test = x_test.reshape(test_length, reshape_pixels)

    # 8 bits per pixel(monochrome)
    x_train /= 8
    x_test /=8

    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)






def compile_model(network, nb_classes, input_shape):
    """Compile a sequential model.

    Args:
        network (dict): the parameters of the network

    Returns:
        a compiled network.

    """
    # Get our network parameters.
    nb_layers = network['nb_layers']
    nb_neurons = network['nb_neurons']
    activation = network['activation']
    optimizer = network['optimizer']

    model = Sequential()

    # Add each layer.
    for i in range(nb_layers):

        # Need input shape for first layer.
        if i == 0:
            model.add(Dense(nb_neurons, activation=activation, input_shape=input_shape))
        else:
            model.add(Dense(nb_neurons, activation=activation))

        model.add(Dropout(0.2))  # hard-coded dropout

    # Output layer.
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])

    return model

def train_and_score(network, dataset):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating

    """
    if dataset == 'cifar10':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_cifar10()
    elif dataset == 'mnist':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_mnist()

    elif dataset == 'tumour_small':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_data_small()


    model = compile_model(network, nb_classes, input_shape)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=10000,  # using early stopping, so no real limit
              verbose=0,
              validation_data=(x_test, y_test),
              callbacks=[early_stopper])

    score = model.evaluate(x_test, y_test, verbose=0)

    return score[1]  # 1 is accuracy. 0 is loss.
