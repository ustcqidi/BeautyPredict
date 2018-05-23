from keras.applications import resnet50
from keras.models import Sequential
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import Dense
from keras.optimizers import Adam
import pickle
import numpy as np
import cv2
import os
from keras.layers import Dropout
#import dlib

resnet = ResNet50(include_top=False, pooling='avg')
model = Sequential()
model.add(resnet)
model.add(Dense(5, activation='softmax'))
model.layers[0].trainable = False

model.load_weights('model-dropout/model-ldl-resnet.h5')

parent_path = os.path.dirname(os.getcwd())
parent_path = os.path.dirname(parent_path)
model_path = parent_path+"/common/haarcascade_frontalface_alt.xml"
data_path = parent_path + "/dataset/SCUT-FBP5500/Images/"
rating_path = parent_path + "/dataset/SCUT-FBP5500/train_test_files/test/test.txt"

image_data = []
attract_data = []
PC = []

score = []
predScore = []


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
        resized_im = cv2.resize(croped_im, (224,224))
    else:
        print(image_name+" error " + str(len(faces)))
    return resized_im

face_cascade = cv2.CascadeClassifier(model_path)

# test_image_data = pickle.load(open('test_image_data.dat','rb'))
test_lable_distribution_data = pickle.load(open('test_lable_distribution.dat','rb'))

data_len = test_lable_distribution_data.__len__()

# test_image = test_image_data[0:data_len]
test_lable_distribution = train_Y = [x for x in test_lable_distribution_data[0:data_len]] #np.array(test_lable_distribution_data[0:data_len])

for i in range(0, data_len):

    label_distribution = test_lable_distribution[i]

    image = label_distribution[1]

    print("测试图片名称:" + str(label_distribution[0]))
    label_score = 1*label_distribution[2][0] + 2*label_distribution[2][1] + 3*label_distribution[2][2] + 4*label_distribution[2][3] + 5*label_distribution[2][4]
    print("标注分数:%1.2f " % (label_score))
    score.append(label_score)

    pred = model.predict(np.expand_dims(image, axis=0))

    ldList = pred[0]
    pred = 1 * ldList[0] + 2 * ldList[1] + 3 * ldList[2] + 4 * ldList[3] + 5 * ldList[4]
    print("预测分数:" + str(pred))
    predScore.append(pred)

y = np.asarray(score)
pred_y = np.asarray(predScore)
corr = np.corrcoef(y, pred_y)[0,1]
print('PC (Pearson correlation) mean = %1.2f ' % (corr))

# APP_ROOT = os.path.dirname(os.path.abspath(__file__))
# parent_path = os.path.dirname(APP_ROOT)
# parent_path = os.path.dirname(parent_path)
# model_path = parent_path+"/common/mmod_human_face_detector.dat"
# cnn_face_detector = dlib.cnn_face_detection_model_v1(model_path)
#
# def beauty_predict(path,img):
#     # print img
#     im0 = cv2.imread(path + "/" + img)
#     if im0 is None:
#         print ('read image error!!')
#         return img
#     if im0.shape[0] > 640:
#         new_shape = (640,im0.shape[1]*640/im0.shape[0])
#     elif im0.shape[1] > 640:
#         new_shape = (im0.shape[0]*640/im0.shape[1],640)
#     elif im0.shape[0] < 320 or im0.shape[1] < 320:
#         new_shape = (im0.shape[0]*2,im0.shape[1]*2)
#     else:
#         new_shape = im0.shape[0:2]
#
#     im = cv2.resize(im0, (int(new_shape[1]), int(new_shape[0])))
#     dets = cnn_face_detector(im, 0)
#     if (len(dets) < 1):
#         cv2.putText(im, 'Face not detect, try again', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
#         ret = path + "/output-" + img
#         cv2.imwrite(ret,im)
#         return "output-"+img
#     for i, d in enumerate(dets):
#         face = [d.rect.left(),d.rect.top(),d.rect.right(),d.rect.bottom()]
#         croped_im = im[face[1]:face[3],face[0]:face[2],:]
#         resized_im = cv2.resize(croped_im, (350,350))
#         # normed_im = np.array([(resized_im-127.5)/127.5])
#         print ('normed image shape:', resized_im.shape)
#         # score = model.predict(resized_im)
#         score = model.predict(np.expand_dims(resized_im, axis=0))
#         if(score == 0.00):
#             info = 'Failed to predict, try again'
#         else:
#             info = str('%.2f'%(score))
#         cv2.rectangle(im, (face[0],face[1]),(face[2],face[3]),(0,255,0),3)
#         cv2.putText(im, info, (face[0],face[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
#     ret = path + "/output-" + img
#     cv2.imwrite(ret,im)
#     return "output-"+img
#
#
# test_image_path = parent_path + "/samples/image"

# beauty_predict(test_image_path, "test1.jpg")
# beauty_predict(test_image_path, "test2.jpg")
# beauty_predict(test_image_path, "test3.jpg")
# beauty_predict(test_image_path, "fengjie.jpg")
# beauty_predict(test_image_path, "test11.jpg")
# beauty_predict(test_image_path, "shunli.jpg")
# beauty_predict(test_image_path, "nenghua.jpg")
