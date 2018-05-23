import cv2
import os, sys
import pickle
import numpy
from os import path

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

parent_path = os.path.dirname(os.getcwd())
parent_path = os.path.dirname(parent_path)

data_path = parent_path + "/dataset/SCUT-FBP5500/Images/"
rating_path = parent_path + "/dataset/SCUT-FBP5500/train_test_files/train/train.txt"
model_path = parent_path + "/common/haarcascade_frontalface_alt.xml"

image_data = []
attract_data = []

rating_file = open(rating_path, 'r')

face_cascade = cv2.CascadeClassifier(model_path)

for line in rating_file.readlines():
    line = line.strip().split(' ')

    im = detectFace(face_cascade, data_path, str(line[0]))

    if isinstance(im, numpy.ndarray):
        normed_im = (im-127.5)/127.5
        image_data.append(normed_im)
        attract_data.append([str(line[0]), float(line[1])])
    else:
        print(str(line[0]) + " 未检测到人脸，丢弃样本")

rating_file.close()

#print attract_data
# for item in attract_data:
    # im = detectFace(face_cascade, data_path, str(item[0]))
    #
    # if isinstance(im, numpy.ndarray):
    #     normed_im = (im-127.5)/127.5
    #     image_data.append(normed_im)
    # else:
    #     print(str(item[0]) + " 未检测到人脸，丢弃样本")
    # imgAbsPath = data_path + str(item[0])
    # img = cv2.imread(imgAbsPath)
    # image_data.append(img)

pickle.dump(image_data, open('image_data.dat','wb'))
pickle.dump(attract_data, open('attractive_data.dat','wb'))