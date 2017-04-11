# -*- coding:utf-8 -*-
import time
from websocket import create_connection
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import base64
from PIL import Image
from io import BytesIO

NUM_CLASSES = 9
IMAGE_SIZE = 224
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*3


host = '127.0.0.1' #お使いのサーバーのホスト名を入れます
port = 8080 #適当なPORTを指定してあげます



ws = create_connection("ws://127.0.0.1:8080/")
while True:
    img = raw_input()
    b64 = open(img,'rb').read().encode('base64')
    ws.send(b64)
    time.sleep(1)
    result =  ws.recv()
    print("Received '%s'" % result)
    time.sleep(1)

ws.close()



