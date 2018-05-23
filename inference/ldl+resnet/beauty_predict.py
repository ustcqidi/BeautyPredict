from keras.layers import Conv2D, Input, MaxPool2D,Flatten, Dense, Permute, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import adam
import numpy as np
import pickle
import keras
import cv2
import sys
import dlib
import os.path
from keras.models import Sequential
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import Dense
from keras.optimizers import Adam
import pickle
import numpy as np
import cv2
import os
from keras.layers import Dropout

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

parent_path = os.path.dirname(APP_ROOT)
parent_path = os.path.dirname(parent_path)
model_path = parent_path+"/common/mmod_human_face_detector.dat"
cnn_face_detector = dlib.cnn_face_detection_model_v1(model_path)

resnet = ResNet50(include_top=False, pooling='avg')
model = Sequential()
model.add(resnet)
model.add(Dense(5, activation='softmax'))
model.layers[0].trainable = False

model.load_weights('model-ldl-resnet.h5')

def score_mapping(modelScore):

    if modelScore <= 1.9:
        mappingScore = ((4 - 2.5) / (1.9 - 1.0)) * (modelScore-1.0) + 2.5
    elif modelScore <= 2.8:
        mappingScore = ((5.5 - 4) / (2.8 - 1.9)) * (modelScore-1.9) + 4
    elif modelScore <= 3.4:
        mappingScore = ((6.5 - 5.5) / (3.4 - 2.8)) * (modelScore-2.8) + 5.5
    elif modelScore <= 4:
        mappingScore = ((8 - 6.5) / (4 - 3.4)) * (modelScore-3.4) + 6.5
    elif modelScore < 5:
        mappingScore = ((9 - 8) / (5 - 4)) * (modelScore-4) + 8

    return mappingScore

def beauty_predict(path, img):
    im0 = cv2.imread(path + "/" + img)

    if im0.shape[0] > 1280:
        new_shape = (1280, im0.shape[1] * 1280 / im0.shape[0])
    elif im0.shape[1] > 1280:
        new_shape = (im0.shape[0] * 1280 / im0.shape[1], 1280)
    elif im0.shape[0] < 640 or im0.shape[1] < 640:
        new_shape = (im0.shape[0] * 2, im0.shape[1] * 2)
    else:
        new_shape = im0.shape[0:2]

    im = cv2.resize(im0, (int(new_shape[1]), int(new_shape[0])))
    dets = cnn_face_detector(im, 0)

    for i, d in enumerate(dets):
        face = [d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom()]
        croped_im = im[face[1]:face[3], face[0]:face[2], :]
        resized_im = cv2.resize(croped_im, (224, 224))
        normed_im = np.array([(resized_im - 127.5) / 127.5])

        pred = model.predict(normed_im)
        ldList = pred[0]
        out = 1 * ldList[0] + 2 * ldList[1] + 3 * ldList[2] + 4 * ldList[3] + 5 * ldList[4]

        out = score_mapping(out)

        print(img + " 打分:" + str('%.2f' % (out)))
        cv2.rectangle(im, (face[0], face[1]), (face[2], face[3]), (0, 255, 0), 3)
        cv2.putText(im, str('%.2f' % (out)), (face[0], face[3]), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

    ret = path + "/output-" + img
    cv2.imwrite(ret, im)
    return ret


# beauty_predict(parent_path+"/samples/image",'fengjie.jpg')
# beauty_predict(parent_path+"/samples/image",'nenghua.jpg')
# beauty_predict(parent_path+"/samples/image",'shunli.jpg')
# beauty_predict(parent_path+"/samples/image",'test1.jpg')
# beauty_predict(parent_path+"/samples/image",'test2.jpg')
# beauty_predict(parent_path+"/samples/image",'test3.jpg')
# beauty_predict(parent_path+"/samples/image",'fty1845.jpg')
beauty_predict(parent_path+"/samples/image",'fty1959.jpg')
# beauty_predict(parent_path+"/samples/image",'jiyou.png')