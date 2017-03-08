# -*- coding: utf-8 -*-

import numpy as np
import cv2
import sys
import csv
from numpy.random import *
import os

def writeImg(frame, n, i):
  if rand() > 0.2:
    image_name = "data/%s/train/%d/%04d.png"  % (sys.argv[1], n, i/4)
  else:
    image_name = "data/%s/test/%d/%04d.png"  % (sys.argv[1], n, i/4)
  cv2.imwrite(image_name, frame)

def Video2Image(items, n):
  try:
    os.mkdir("data/%s/train/%d"  % (sys.argv[1], n))
  except:
    pass
  try:
    os.mkdir("data/%s/test/%d"  % (sys.argv[1], n))
  except:
    pass
  print("start %s" % (items[n],))
  video = 'data/'+ sys.argv[1] + '/' + str(n) + '.m4v'
  cap = cv2.VideoCapture(video)
  i = 0
  item_list = []
  while(cap.isOpened()):
      ret, frame = cap.read()
      i +=1
      if(i % 2 == 0):
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
    os.mkdir('data/' + sys.argv[1] + '/train')
  except:
    pass
  try:
    os.mkdir('data/' + sys.argv[1] + '/test')
  except:
    pass
  try:
    f = open('data/'+ sys.argv[1] + "/list.csv", 'r')
    items = [item[0] for item in csv.reader(f)]
    f.close()
  except IOError:
    print("ファイルが存在しません")
    exit()

  for n in range(len(items)):
    Video2Image(items, n)




if __name__ == '__main__':
  main()
