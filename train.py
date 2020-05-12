from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from PIL import Image
import numpy as np
import cv2
import csv
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

# Helper: Early stopping.
early_stopper = EarlyStopping(patience=5)

# smaller variant of the brain tumour dataset
def get_data(data_type):
    nb_classes = 2
    batch_size = 100
    # input_shape = (128*128,)
    input_shape = (128,128,1)


    def fetch_image(item):
        image = Image.open(item)
        # convert image to monochrome
        image = image.convert('L')
        # convert image to numpy array
        data = np.asarray(image)
        # print(data.shape)
        return data

    x_train = []
    y_train = []

    x_test = []
    y_test = []

    with open('meta/{0}/train.csv'.format(data_type), 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            x_train.append(row[0])
            y_train.append(row[1])

    with open('meta/{0}/test.csv'.format(data_type), 'r') as file:
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
    # reshape_pixels = 128*128
    # x_train = x_train.reshape(train_length, reshape_pixels)
    # x_test = x_test.reshape(test_length, reshape_pixels)

    # 8 bits per pixel(monochrome)
    x_train /= 8
    x_test /=8

    # New axis for 2d
    x_train = x_train[...,np.newaxis]
    x_test = x_test[...,np.newaxis]

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
    nb_dense_layers = network['nb_dense_layers']
    nb_neurons = network['nb_neurons']
    activation = network['activation']
    optimizer = network['optimizer']
    dropout_rate = network['dropout']
    filters = network['filters']
    kernel_size = network['kernel_sizes']

    model = Sequential()
    # Add each layer.
    for i in range(nb_layers):
        # Need input shape for first layer.
        if i == 0:
            model.add(Conv2D(filters, kernel_size, activation=activation, padding="same", input_shape=input_shape))
        else:
            model.add(Conv2D(filters, kernel_size, activation=activation, padding="same"))
        
        model.add(MaxPooling2D(padding="same", pool_size=(2,2)))
    
    # Flatten
    model.add(Flatten())   

    for i in range(nb_dense_layers):
        if i ==0:
            model.add(Dense(nb_neurons, activation=activation, input_shape=input_shape))
            model.add(Dropout(dropout_rate)) 

        elif i<nb_dense_layers-1:
            model.add(Dense(nb_neurons, activation=activation))
            model.add(Dropout(dropout_rate)) 

        else:
            model.add(Dense(nb_neurons, activation=activation))
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
    nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_data(dataset)

    model = compile_model(network, nb_classes, input_shape)

    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=1000,  # using early stopping, so no real limit
              verbose=0,
              validation_data=(x_test, y_test),
              callbacks=[early_stopper])

    score = model.evaluate(x_test, y_test, verbose=0)
    return score[1], history.history['loss'],history.history['val_loss']
    # return score  # 1 is accuracy. 0 is loss
