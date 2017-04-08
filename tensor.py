# -*- coding: utf-8 -*-


from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Convolution2D, Flatten, MaxPooling2D, Dropout, Input
from keras import optimizers
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import ResNet50
import keras.backend.tensorflow_backend as KTF
import sys
import numpy as np
import os, os.path
import csv
from tensorflow.python.framework import graph_util

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from tensorflow.python.framework import graph_util

old_session = KTF.get_session()
sess = tf.Session()
KTF.set_session(sess)

data_directory = 'treasure'
bottom_model_name = 'resnet'

f = open('./data/' + data_directory + '/list.csv','r')
items = [item[0] for item in csv.reader(f)]
f.close()
NUM_CLASSES = len(items)
IMAGE_SIZE = 224
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*3

model_filename = 'fine_metdata.json'
weights_filename = 'fine.hdf5'

if __name__ == '__main__':
  with tf.Graph().as_default() as graph:
      with tf.Session() as sess:

        input_tensor = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

        bottom_model = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)


        top_model = Sequential()
        top_model.add(Flatten(input_shape=bottom_model.output_shape[1:]))
        top_model.add(Dense(1024, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(NUM_CLASSES, activation='softmax'))

        model = Model(input=bottom_model.input, output=top_model(bottom_model.output))
        model.load_weights(os.path.join('./data/' + data_directory + '/model/' + bottom_model_name + '_withCNN2.h5'))
        model.save("fine.hdf5")

        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        graph_def = graph_util.convert_variables_to_constants(
          sess, graph.as_graph_def(), ['init_2'])
        tf.train.write_graph(graph_def=graph_def, logdir='.', name='model2.pb', as_text=False)
        tf.train.write_graph(graph_def=graph_def, logdir='.', name='model2.txt', as_text=True)
        saver = tf.train.Saver()
        saver.save(sess, 'model2.ckpt')

