# -*- coding:utf-8 -*-
from websocket_server import WebsocketServer


import numpy as np
from keras.models import load_model
from keras import optimizers
import keras.backend.tensorflow_backend as KTF
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import os.path
import sys
import base64
from PIL import Image
from io import BytesIO
import csv


host = '127.0.0.1' #お使いのサーバーのホスト名を入れます
port = 8080 #クライアントと同じPORTをしてあげます

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


with tf.Graph().as_default():
    session = tf.Session('')
    KTF.set_session(session)

    json_string = open(model_filename).read()
    model = load_model(model_filename)


    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])
    model.summary()
    input_image = np.expand_dims(img_to_array(load_img('0000.jpg', target_size=(IMAGE_SIZE,IMAGE_SIZE)))/255,axis=0)
    result = np.argmax(model.predict(input_image, verbose=1))


    print('waiting')

    def new_client(client, server):
        server.send_message_to_all("Hey all, a new client has joined us")

    def send_msg_allclient(client, server,message):
        print(message)
        try:
          im = img_to_array(Image.open(BytesIO(base64.b64decode(message))).resize((IMAGE_SIZE,IMAGE_SIZE)))/255
          input_image = np.expand_dims(im,axis=0)
          result = np.argmax(model.predict(input_image, verbose=1))

          server.send_message_to_all(items[result])
        except:
          server.send_message_to_all("エラー")

    server = WebsocketServer(port, host=host)
    server.set_fn_message_received(send_msg_allclient)
    server.run_forever()


