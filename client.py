# -*- coding:utf-8 -*-
import time
from websocket import create_connection
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

NUM_CLASSES = 9
IMAGE_SIZE = 224
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*3


host = '127.0.0.1' #お使いのサーバーのホスト名を入れます
port = 8080 #適当なPORTを指定してあげます


ws = create_connection("ws://127.0.0.1:8080/")

while True:
    ws.send("Hello, World")
    time.sleep(1)
    result =  ws.recv()
    print("Received '%s'" % result)
    time.sleep(1)

ws.close()

img = raw_input()
input_image = img_to_array(load_img(img, target_size=(IMAGE_SIZE,IMAGE_SIZE)))/255
client.send(input_image) #適当なデータを送信します（届く側にわかるように）
response = client.recv(1024) #レシーブは適当な2進数にします（大きすぎるとダメ）
print response

