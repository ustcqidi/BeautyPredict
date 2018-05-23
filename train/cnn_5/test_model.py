from keras.layers import Conv2D, Input, MaxPool2D,Flatten, Dense, Permute
from keras.models import Model
from keras.optimizers import adam
import numpy as np
import pickle
import keras
import cv2
import os
import dlib


def detectFace(detector,image_path, image_name):
    imgAbsPath = image_path + image_name
    img = cv2.imread(imgAbsPath)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    w = img.shape[1]
    faces = detector.detectMultiScale(gray, 1.1,5,0,(w//2,w//2))

    resized_im = 0

    if len(faces) == 1:
        face = faces[0]
        croped_im = img[face[1]:face[1]+face[3],face[0]:face[0]+face[2],:]
        resized_im = cv2.resize(croped_im, (128,128))
        cv2.imwrite("../data/"+image_name, resized_im)
    else:
        print(image_name+" error " + str(len(faces)))
    return resized_im

input = Input(shape=[128,128,3])
x = Conv2D(50,(5,5), strides=1, padding='valid',name='conv1',activation='relu')(input)
x = MaxPool2D(pool_size=3,strides=2, padding='same')(x)
x = Conv2D(100,(5,5), strides=1, padding='valid',name='conv2',activation='relu')(x)
x = MaxPool2D(pool_size=3,strides=2, padding='same')(x)
x = Conv2D(150,(4,4), strides=1, padding='valid',name='conv3',activation='relu')(x)
x = MaxPool2D(pool_size=3,strides=2, padding='same')(x)
x = Conv2D(200,(4,4), strides=1, padding='valid',name='conv4',activation='relu')(x)
x = MaxPool2D(pool_size=3,strides=2, padding='same')(x)
x = Conv2D(250,(4,4), strides=1, padding='valid',name='conv5',activation='relu')(x)
x = MaxPool2D(pool_size=3,strides=2, padding='same')(x)
# x = Conv2D(300,(2,2), strides=1, padding='valid',name='conv6',activation='relu')(x)
# x = MaxPool2D(pool_size=3,strides=2, padding='same')(x)
x = Flatten()(x)
x = Dense(128, name='dense1')(x)
score = Dense(1,name='score')(x)

model = Model([input], [score])
#print model.summary()
model.load_weights('model-dropout/model-50.h5', by_name=True)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

parent_path = os.path.dirname(APP_ROOT)
parent_path = os.path.dirname(parent_path)
model_path = parent_path+"/common/haarcascade_frontalface_alt.xml"

data_path = parent_path + "/dataset/SCUT-FBP5500/Images/"
rating_path = parent_path + "/dataset/SCUT-FBP5500/train_test_files/test/test.txt"

image_data = []
attract_data = []
PC = []

score = []
predScore = []

face_cascade = cv2.CascadeClassifier(model_path)
rating_file = open(rating_path, 'r')

for line in rating_file.readlines():
    line = line.strip().split(' ')
    attract_data.append([str(line[0]), float(line[1])])

rating_file.close()

# read img
for item in attract_data:

    im = detectFace(face_cascade, data_path, str(item[0]))

    if isinstance(im, np.ndarray):
        normed_im = (im - 127.5) / 127.5

        pred = model.predict(np.expand_dims(normed_im, axis=0))

        print("测试图片名称:" + str(item[0]))
        print("标注分数:" + str(item[1]))
        print("预测结果:" + str('%.2f' % (pred)))

        score.append(item[1])
        pred = min(pred[0])
        predScore.append(pred)
    else:
        print(str(item[0]) + " 未检测到人脸，丢弃样本")

y = np.asarray(score)
pred_y = np.asarray(predScore)
corr = np.corrcoef(y, pred_y)[0,1]
print('PC (Pearson correlation) mean = %1.2f ' % (corr))

# for img in test_imgs:
#     im0 = cv2.imread(test_image_path + "/" + img)
#     if im0 is None:
#         print('read image error!!')
#
#     if im0.shape[0] > 640:
#         new_shape = (640, im0.shape[1] * 640 / im0.shape[0])
#     elif im0.shape[1] > 640:
#         new_shape = (im0.shape[0] * 640 / im0.shape[1], 640)
#     elif im0.shape[0] < 320 or im0.shape[1] < 320:
#         new_shape = (im0.shape[0] * 2, im0.shape[1] * 2)
#     else:
#         new_shape = im0.shape[0:2]
#
#     im = cv2.resize(im0, (int(new_shape[1]), int(new_shape[0])))
#     dets = cnn_face_detector(im, 0)
#     if (len(dets) < 1):
#         cv2.putText(im, 'Face not detect, try again', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         ret = test_image_path + "/output-" + img
#         cv2.imwrite(ret, im)
#
#     for i, d in enumerate(dets):
#         face = [d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom()]
#         croped_im = im[face[1]:face[3], face[0]:face[2], :]
#         resized_im = cv2.resize(croped_im, (128, 128))
#         normed_im = np.array([(resized_im - 127.5) / 127.5])
#         # print('normed image shape:', normed_im.shape)
#         score = model.predict(normed_im)
#
#         print(score)