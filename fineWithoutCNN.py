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



f = open('./data/' + sys.argv[1] + '/list.csv','r')
items = [item[0] for item in csv.reader(f)]
f.close()
NUM_CLASSES = len(items)
IMAGE_SIZE = 224
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*3
TRAIN_DATA = './data/' + sys.argv[1] +'/train'
TEST_DATA = './data/' + sys.argv[1] +'/test'


if __name__ == '__main__':
    if not os.path.exists('./data/' + sys.argv[1] + '/model'):
        os.mkdir('./data/' + sys.argv[1] + '/model')
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        dim_ordering='tf')
    train_data = train_datagen.flow_from_directory(
        TRAIN_DATA,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=20,
        classes=map(str, range(NUM_CLASSES)))

    test_datagen = ImageDataGenerator(
        rescale=1./255,
        dim_ordering='tf')
    test_data = test_datagen.flow_from_directory(
        TEST_DATA,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=20,
        classes=map(str, range(NUM_CLASSES)))


    input_tensor = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    vgg_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)


    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg_model.output_shape[1:]))
    top_model.add(Dense(1024, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(NUM_CLASSES, activation='softmax'))

    model = Model(input=vgg_model.input, output=top_model(vgg_model.output))

    for layer in model.layers:
        layer.trainable = False

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])
    checkpoint = ModelCheckpoint('./data/' + sys.argv[1] + '/model/model4.h5', monitor='val_acc', verbose=1, save_best_only=False)
    model.fit_generator(
        train_data,
        samples_per_epoch=100,
        nb_epoch=50,
        callbacks=[checkpoint],
        validation_data=test_data,
        nb_val_samples=100)




