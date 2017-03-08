# -*- coding: utf-8 -*-



from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Convolution2D, Flatten, MaxPooling2D, Dropout, Input
from keras.utils.np_utils import to_categorical
from keras import optimizers
from keras.layers import InputLayer
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import sys
import numpy as np
import os, os.path
import csv
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3

data_directory = 'treasure'
bottom_model_name = sys.argv[1]

f = open('./data/' + data_directory + '/list.csv','r')
items = [item[0] for item in csv.reader(f)]
f.close()
NUM_CLASSES = len(items)
IMAGE_SIZE = 224
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*3
TEST_DATA = './data/' + data_directory +'/image'


if __name__ == '__main__':
    if not os.path.exists('./data/' + data_directory + '/model'):
        os.mkdir('./data/' + data_directory + '/model')

    test_datagen = ImageDataGenerator(
        rescale=1./255,
        dim_ordering='tf')
    test_data = test_datagen.flow_from_directory(
        TEST_DATA,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=20,
        classes=map(str, range(NUM_CLASSES)))

    input_tensor = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    if bottom_model_name == 'vgg':
        bottom_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
    elif(bottom_model_name == 'resnet'):
        bottom_model = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)
    elif(bottom_model_name == 'inception'):
        bottom_model = InceptionV3(include_top=False, weights='imagenet', input_tensor=input_tensor)

    top_model = Sequential()
    top_model.add(Flatten(input_shape=bottom_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(NUM_CLASSES, activation='softmax'))

    model = Model(input=bottom_model.input, output=top_model(bottom_model.output))
    model.load_weights(os.path.join('./data/' + data_directory + '/model/' + bottom_model_name + '_withoutCNN.h5'))
    for layer in model.layers:
        layer.trainable = False
    for layer in top_model.layers:
        layer.trainable = False

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])
    model.fit_generator(
        test_data,
        samples_per_epoch=20,
        nb_epoch=1,
        validation_data=test_data,
        nb_val_samples=302)


