# -*- coding: utf-8 -*-

import numpy as np
import cv2
import sys
import csv
from numpy.random import *
import os
import zipfile

data_directory = 'treasure'
per_image = 10

def writeImg(frame, n, i):
  image_name = "data/%s/train/%d/%04d.png"  % (data_directory, n, i/per_image)

  cv2.imwrite(image_name, cv2.resize(frame, (320,320)))

def Video2Image(items, n):
  try:
    os.mkdir("data/%s/train/%d"  % (data_directory, n))
  except:
    pass
  print("start %s" % (items[n],))
  video = 'data/'+ data_directory + '/' + str(n) + '.m4v'
  cap = cv2.VideoCapture(video)
  i = 0
  item_list = []
  while(cap.isOpened()):
      ret, frame = cap.read()
      i +=1
      if(i % per_image == 0):
        writeImg(frame, n, i)
      if(i%100==0):
        print(i)
      if(cap.get(cv2.CAP_PROP_POS_AVI_RATIO) == 1):
        break
  cap.release()
  cv2.destroyAllWindows()
  print("finish %s" % (items[n],))


def main():
  try:
    os.mkdir("data/%s/train/"  % (data_directory))
  except:
    pass
  try:
    f = open('data/'+ data_directory + "/list.csv", 'r')
    items = [item[0] for item in csv.reader(f)]
    f.close()
  except IOError:
    print("ファイルが存在しません")
    exit()

  for n in range(len(items)):
    Video2Image(items, n)




if __name__ == '__main__':
  main()
