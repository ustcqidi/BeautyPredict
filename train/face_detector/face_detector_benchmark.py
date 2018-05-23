import cv2
import os
import dlib
from mtcnn.mtcnn import MTCNN

parent_path = os.path.dirname(os.getcwd())
parent_path = os.path.dirname(parent_path)
data_path = parent_path + "/dataset/SCUT-FBP5500/Images/"
rating_path = parent_path + "/dataset/SCUT-FBP5500/train_test_files/train/train.txt"
opencv_model_path = parent_path + "/common/haarcascade_frontalface_alt.xml"
dlib_model_path = parent_path + "/common/mmod_human_face_detector.dat"
cornercase_path_single = parent_path + "/dataset/face_test/single_face/"
cornercase_path_no = parent_path + "/dataset/face_test/no_face/"
cornercase_path_multi = parent_path + "/dataset/face_test/multi_face/"

image_data = []
attract_data = []
rating_file = open(rating_path, 'r')
# 人脸检测器
print("face detector benchmark start~")
face_cascade_detector = cv2.CascadeClassifier(opencv_model_path)
dlib_face_detector = dlib.cnn_face_detection_model_v1(dlib_model_path)
mtcnn_detector = MTCNN()

errors = [0, 0, 0, 0]


def detectFace(image_path, image_name):
    imgAbsPath = image_path + image_name
    print(imgAbsPath)
    img = cv2.imread(imgAbsPath)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    w = img.shape[1]

    facesCV = face_cascade_detector.detectMultiScale(gray, 1.1, 5, 0, (w // 2, w // 2))
    if len(facesCV) < 1:
        errors[0] = errors[0] + 1
        print(image_name + " opencv error " + str(len(facesCV)))

    pic = cv2.equalizeHist(gray)
    facesCVEqual = face_cascade_detector.detectMultiScale(pic, 1.1, 5, 0, (w // 2, w // 2))
    if len(facesCVEqual) < 1:
        errors[1] = errors[1] + 1
        print(image_name + " opencv equal error " + str(len(facesCVEqual)))

    facesMTCNN = mtcnn_detector.detect_faces(img)
    print(facesMTCNN)
    if len(facesMTCNN) < 1:
        errors[2] = errors[2] + 1
        print(image_name + " mtcnn error " + str(len(facesMTCNN)))
    if len(facesMTCNN) == 1:
        box = facesMTCNN[0]['box']
        cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 155, 255), 2)
        cv2.imwrite(image_path + "/mtcnn_" + image_name, img)
    else:
        for mtcnn in facesMTCNN:
            box = mtcnn['box']
            cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 155, 255), 2)
        cv2.imwrite(image_path + "/mtcnn_" + image_name, img)

    facesDlib = dlib_face_detector(gray, 0)
    if len(facesDlib) < 1:
        errors[3] = errors[3] + 1
        print(image_name + " dlib error " + str(len(facesDlib)))
    else:
        print(facesDlib)
    return errors


def testBaseDataSet():
    for line in rating_file.readlines():
        line = line.strip().split(' ')
        detectFace(data_path, str(line[0]))

    rating_file.close()

    print("opencv错误个数（灰度化）= " + str(errors[0]))
    print("opencv错误个数（使用直方图均衡化） = " + str(errors[1]))
    print("mtcnn错误个数（原图） = " + str(errors[2]))
    print("dlib错误个数（灰度化）=" + str(errors[3]))


def test1FaceCase():
    pathDir = os.listdir(cornercase_path_single)
    for dir in pathDir:
        if (dir.endswith(".jpg")):
            detectFace(cornercase_path_single, dir)

    print("opencv错误个数（灰度化）= " + str(errors[0]))
    print("opencv错误个数（使用直方图均衡化） = " + str(errors[1]))
    print("mtcnn错误个数（原图） = " + str(errors[2]))
    print("dlib错误个数（灰度化）=" + str(errors[3]))
    # result = 20,20,7,18 mtcnn优势明显


def testNoFaceCase():
    pathDir = os.listdir(cornercase_path_no)
    for dir in pathDir:
        if (dir.endswith(".jpg")):
            detectFace(cornercase_path_no, dir)

    print("opencv错误个数（灰度化）= " + str(errors[0]))
    print("opencv错误个数（使用直方图均衡化） = " + str(errors[1]))
    print("mtcnn错误个数（原图） = " + str(errors[2]))
    print("dlib错误个数（灰度化）=" + str(errors[3]))
    # result = 17,17,17,17 全部通过


def testMultiFaceCase():
    pathDir = os.listdir(cornercase_path_multi)
    for dir in pathDir:
        if (dir.endswith(".jpg")):
            detectFace(cornercase_path_multi, dir)

    print("opencv错误个数（灰度化）= " + str(errors[0]))
    print("opencv错误个数（使用直方图均衡化） = " + str(errors[1]))
    print("mtcnn错误个数（原图） = " + str(errors[2]))
    print("dlib错误个数（灰度化）=" + str(errors[3]))
    # mctnn结果最好，画的框也比较准确


# testBaseDataSet()
test1FaceCase()
# testNoFaceCase()
# testMultiFaceCase()
# detectFace(data_path, "mty320.jpg")
# fty849.jpg mty394.jpg mty1329.jpg
