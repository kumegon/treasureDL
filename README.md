Treasure Learning
====

## Overview
We can train treasures' images using this script.

##

## <a name ="req"> Requirement</a>
* Python 2.7.10
* OpenCV
* numpy
* tensorflow 0.10.0
* Keras

## Usage
1. Please download [the data](https://drive.google.com/open?id=0BwGWkWRitjNiQ01LazBUeU1TMWs), unzip, and move to the this script's directory.

2. Capture a frame from a video clip
    1. Please start 'rippingVideo.py'.
        ```
        $python ripping.py
        ```

3. Starting learning.
    1. I wrote two programs, fineWithCNN.py, fineWithoutCNN.py.
    2. You can start them like this(you can select those models, vgg, resnet, inception.
        ```
        $python fineWithCNN.py vgg
        ```


## Install
Please read [Requirement](#req) section, and install packages you haven't install.


## Author

[kumegon](https://github.com/kumegon)
