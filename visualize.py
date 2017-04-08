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
import numpy as np
import os, os.path
import csv
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.utils.visualize_util import plot



IMAGE_SIZE = 224



#if __name__ == '__main__':

    '''
    bottom_model_name = sys.argv[1]
    input_tensor = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    if bottom_model_name == 'vgg':
        bottom_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
    elif(bottom_model_name == 'resnet'):
        bottom_model = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)
    elif(bottom_model_name == 'inception'):
        bottom_model = InceptionV3(include_top=False, weights='imagenet', input_tensor=input_tensor)
    '''
    input_tensor = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    bottom_model = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)


    top_model = Sequential()
    top_model.add(Flatten(input_shape=bottom_model.output_shape[1:]))
    top_model.add(Dense(1024, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(9, activation='softmax'))

    model = Model(input=bottom_model.input, output=top_model(bottom_model.output))
    #plot(model, to_file='model.png')
    #model.summary()
    print(model.layers[0])
    print(model.layers[-1].name)





