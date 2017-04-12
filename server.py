# -*- coding: utf-8 -*-
import tornado.ioloop
import tornado.web
import tornado.websocket
from tornado.options import define, options ,parse_command_line
from datetime import datetime
import shlex, subprocess , time ,re , json ,threading



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



data_directory = 'treasure'
bottom_model_name = 'resnet'

f = open('./data/' + data_directory + '/list.csv','r')
items = [item[0] for item in csv.reader(f)]
f.close()
NUM_CLASSES = len(items)

IMAGE_SIZE = 224
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*3

model_filename = './fine.hdf5'


IP = '54.199.224.250'


#WebSocketがListenするポートを指定
define("port", default = 8080,type = int)

class SendWebSocket(tornado.websocket.WebSocketHandler):

    #コネクションが確保されると呼び出されるイベント
    def open(self):
        print 'Session Opened. IP:' + self.request.remote_ip
        self.write_message("Hey all, a new client has joined us")

    #ブラウザが閉じられた場合等，切断イベントが発生した場合のイベント
    def on_close(self):
        print "Session closed"

    #クライアントからメッセージが送られてくると呼び出されるイベント
    def on_message(self, message):
      try:
        im = img_to_array(Image.open(BytesIO(base64.b64decode(message))).resize((IMAGE_SIZE,IMAGE_SIZE)))/255
        input_image = np.expand_dims(im,axis=0)
        result = np.argmax(model.predict(input_image, verbose=1))
        self.write_message(items[result])
      except:
        self.write_message("エラー")

    #Trueにしないと明示されたホストからしか通信を受け付けない
    def check_origin(self, origin):
        return True



old_session = KTF.get_session()


with tf.Graph().as_default():
    session = tf.Session('')
    KTF.set_session(session)

    json_string = open(model_filename).read()
    model = load_model(model_filename)


    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])
    '''
    model.summary()
    input_image = np.expand_dims(img_to_array(load_img('0000.jpg', target_size=(IMAGE_SIZE,IMAGE_SIZE)))/255,axis=0)
    result = np.argmax(model.predict(input_image, verbose=1))
    '''
    print('waiting')

    app = tornado.web.Application([
      (r"/", SendWebSocket),
    ])

    parse_command_line()
    app.listen(options.port)
    mainloop = tornado.ioloop.IOLoop.instance()
    mainloop.start() #WebSocketServer起動



