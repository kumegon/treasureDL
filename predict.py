# -*- coding: utf-8 -*-


import numpy as np
from keras.models import model_from_json
from keras.utils import np_utils
from keras import optimizers
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import os.path
import sys

data_directory = 'treasure'
bottom_model_name = 'resnet'

f = open('./data/' + data_directory + '/list.csv','r')
items = [item[0] for item in csv.reader(f)]
f.close()
NUM_CLASSES = len(items)

IMAGE_SIZE = 224
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*3

model_filename = './fine.hdf5'



old_session = KTF.get_session()

input_images = np.asarray(map(lambda x: img_to_array(load_img(x, target_size=(IMAGE_SIZE,IMAGE_SIZE)))/255 , sys.argv[1:]))

with tf.Graph().as_default():
    session = tf.Session('')
    KTF.set_session(session)

    json_string = open(model_filename).read()
    model = keras.models.load_model(model_filename)


    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    #model.load_weights(weights_filename)

    result = np.argmax(model.predict(input_images, verbose=1))

    results = map(np.argmax, model.predict(input_images, verbose=1))
    for i in range(len(results)):
        print(sys.argv[1+i] + ':' + str(results[i]))

