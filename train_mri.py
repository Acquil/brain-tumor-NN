from PIL import Image
import numpy as np
import cv2
import csv
from keras.utils.np_utils import to_categorical

def get_data_small():
    nb_classes = 2
    batch_size = 10
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

