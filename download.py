# -*- coding: utf-8 -*-
data_directory = 'treasure'

from time import sleep
import sys
import requests
#import urllib2
import urllib
import csv
import os
from PIL import Image
from PIL import ImageOps

def download(image_url, i):
    # 画像ソースを開く
    try:
        response = urllib.urlopen(image_url[0])
        # HTTPステータスコードが200でないなら0を返す
        if response.code != 200:
            return -1
        im = Image.open(response)

        image_type = image_url[0].split(".")[-1]
        image_name = "data/%s/image/%s/%04d.%s" % (data_directory, image_url[1], i, image_type,)
        im.save(image_name, quality=100, optimize=True)
        print(image_name)
    except Exception as e:
        print(e)

if __name__ == '__main__':
    try:
        os.mkdir('data/' + data_directory + '/image')
    except:
        pass
    try:
        f = open('data/'+ data_directory + "/test.csv", 'r')
        items = [item for item in csv.reader(f)]
        f.close()
    except IOError:
        print("ファイルが存在しません")
        exit()

    for i in range(len(items)):
        try:
            os.mkdir("data/%s/image/%s"  % (data_directory, items[i][1]))
        except:
            pass
        download(items[i],i)

