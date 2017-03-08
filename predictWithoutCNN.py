# -*- coding: utf-8 -*-



from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Convolution2D, Flatten, MaxPooling2D, Dropout, Input
from keras.utils.np_utils import to_categorical
from keras import optimizers
from keras.layers import InputLayer
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import sys
import cv2
import numpy as np
import os, os.path
import csv
from keras.applications.vgg16 import VGG16

data_directory = 'treasure'

f = open('./data/' + data_directory + '/list.csv','r')
items = [item[0] for item in csv.reader(f)]
f.close()
NUM_CLASSES = len(items)
IMAGE_SIZE = 224
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*3
TRAIN_DATA = './data/' + data_directory +'/train'
TEST_DATA = './data/' + data_directory +'/test'


if __name__ == '__main__':
    if not os.path.exists('./data/' + data_directory + '/model'):
        os.mkdir('./data/' + data_directory + '/model')
    input_images = np.asarray(map(lambda x: cv2.resize(cv2.imread(x), (IMAGE_SIZE, IMAGE_SIZE))/255  , sys.argv[1:]))


    input_tensor = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    vgg_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
    vgg_modela = VGG16(include_top=True, weights='imagenet', input_tensor=input_tensor)


    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(NUM_CLASSES, activation='softmax'))

    model = Model(input=vgg_model.input, output=top_model(vgg_model.output))
    model.load_weights(os.path.join('./data/' + data_directory + '/model/model3.h5'))


    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    results = map(np.argmax, model.predict(input_images, verbose=1))
    print(results)
    for i in range(len(results)):
        print(sys.argv[2+i] + ':' + items[results[i]])


