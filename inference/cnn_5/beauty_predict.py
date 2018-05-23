from keras.layers import Conv2D, Input, MaxPool2D,Flatten, Dense, Permute
from keras.models import Model
from keras.optimizers import adam
import numpy as np
import pickle
import keras
import cv2
import sys
import dlib
#from PIL import Image, ImageDraw, ImageFont
import os
from os import path

input = Input(shape=[128,128,3])
x = Conv2D(30,(5,5), strides=1, padding='valid',name='conv1',activation='relu')(input)
x = MaxPool2D(pool_size=3,strides=2, padding='same')(x)
x = Conv2D(50,(5,5), strides=1, padding='valid',name='conv2',activation='relu')(x)
x = MaxPool2D(pool_size=3,strides=2, padding='same')(x)
x = Conv2D(80,(5,5), strides=1, padding='valid',name='conv3',activation='relu')(x)
x = MaxPool2D(pool_size=3,strides=2, padding='same')(x)
x = Conv2D(100,(5,5), strides=1, padding='valid',name='conv4',activation='relu')(x)
x = MaxPool2D(pool_size=3,strides=2, padding='same')(x)
x = Flatten()(x)
x = Dense(64, name='dense1',activation='relu')(x)
score = Dense(1,name='score')(x)

model = Model([input], [score])
#print model.summary()
model.load_weights('model-12.h5', by_name=True)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

parent_path = os.path.dirname(APP_ROOT)
parent_path = os.path.dirname(parent_path)
model_path = parent_path+"/common/mmod_human_face_detector.dat"
cnn_face_detector = dlib.cnn_face_detection_model_v1(model_path)

def beauty_predict(path,img):
    im0 = cv2.imread(path + "/" + img)

    if im0.shape[0] > 1280:
        new_shape = (1280,im0.shape[1]*1280/im0.shape[0])
    elif im0.shape[1] > 1280:
        new_shape = (im0.shape[0]*1280/im0.shape[1],1280)
    elif im0.shape[0] < 640 or im0.shape[1] < 640:
        new_shape = (im0.shape[0]*2,im0.shape[1]*2)
    else:
        new_shape = im0.shape[0:2]

    im = cv2.resize(im0, (int(new_shape[1]),int(new_shape[0])))
    dets = cnn_face_detector(im, 0)
    
    for i, d in enumerate(dets):
        face = [d.rect.left(),d.rect.top(),d.rect.right(),d.rect.bottom()]
        croped_im = im[face[1]:face[3],face[0]:face[2],:]
        resized_im = cv2.resize(croped_im, (128,128))
        normed_im = np.array([(resized_im-127.5)/127.5])
        out = model.predict(normed_im)
        cv2.rectangle(im, (face[0],face[1]),(face[2],face[3]),(0,255,0),3)
        cv2.putText(im, str('%.2f'%(min(out[0,0]*2.0+2.0,10.0))), (face[0],face[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
        cv2.putText(im, str('%.2f'%(d.confidence)), (face[0],face[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),2)
        # print out[0,0]

    ret = path + "/output-" + img
    cv2.imwrite(ret,im)
    return ret

beauty_predict(parent_path+"/samples/image",'fengjie.jpg')