import os
import dlib
import cv2
from mtcnn.mtcnn import MTCNN

parent_path = os.path.dirname(os.getcwd())
parent_path = os.path.dirname(parent_path)
dlib_model_path = parent_path + "/common/mmod_human_face_detector.dat"

dlib_face_detector = dlib.cnn_face_detection_model_v1(dlib_model_path)
mtcnn_detector = MTCNN()


def detectFace(image_full_path):
    print(image_full_path)
    img = cv2.imread(image_full_path)

    faces = []
    facesMTCNN = mtcnn_detector.detect_faces(img)
    if len(facesMTCNN) < 1:
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        dets = dlib_face_detector(gray, 1)
        for i, d in enumerate(dets):
            # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(
            #     i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))
            faces.append([d.rect.left(), d.rect.top(), d.rect.right() - d.rect.left(), d.rect.bottom() - d.rect.top()])
        return faces
    if len(facesMTCNN) == 1:
        # print(facesMTCNN)
        faces.append(facesMTCNN[0]['box'])
        return faces
        # cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 155, 255), 2)
        # cv2.imwrite(image_path + "/mtcnn_" + image_name, img)
    else:
        # print(facesMTCNN)
        for mtcnn in facesMTCNN:
            if mtcnn['confidence'] > 0.79:
                faces.append(mtcnn['box'])
        return faces


def test(image_path, image_name):
    print(detectFace(image_path + image_name))


def testAll(image_path):
    error = 0
    pathDir = os.listdir(image_path)
    for file in pathDir:
        if file.endswith(".jpg"):
            list = detectFace(image_path + file)
            if len(list) == 0:
                error = error + 1
    print("error case = " + str(error))


test(parent_path + "/dataset/SCUT-FBP5500/Images/", "mty320.jpg")
test(parent_path + "/dataset/SCUT-FBP5500/Images/", "fty849.jpg")
test(parent_path + "/dataset/face_test/single_face/", "output-tmp_3c39_2018-04-13-18-16-26.jpg")
test(parent_path + "/dataset/face_test/no_face/", "output-tmp_879a_2018-04-09-10-48-38.jpg")
test(parent_path + "/dataset/face_test/multi_face/", "output-tmp_63de_2018-04-13-14-38-56.jpg")
test(parent_path + "/dataset/face_test/multi_face/", "output-tmp_537c_2018-04-17-22-30-30.jpg")
testAll(parent_path + "/dataset/face_test/single_face/")
