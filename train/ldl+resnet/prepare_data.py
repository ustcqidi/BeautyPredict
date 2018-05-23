import cv2
import os, sys
import pickle
import numpy as np
import numpy
from os import path
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import random
from PIL import Image
from PIL import ImageEnhance
import scipy.misc

parent_path = os.path.dirname(os.getcwd())
parent_path = os.path.dirname(parent_path)
data_path = parent_path + "/dataset/SCUT-FBP5500/Images/"
rating_path = parent_path + "/dataset/SCUT-FBP5500/All_Ratings/"
model_path = parent_path + "/common/haarcascade_frontalface_alt.xml"
face_cascade = cv2.CascadeClassifier(model_path)

# datagen = ImageDataGenerator(
#         rotation_range=0.2,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         fill_mode='nearest')
datagen = ImageDataGenerator(
                 featurewise_center = False,             #是否使输入数据去中心化（均值为0），
                 samplewise_center  = False,             #是否使输入数据的每个样本均值为0
                 featurewise_std_normalization = False,  #是否数据标准化（输入数据除以数据集的标准差）
                 samplewise_std_normalization  = False,  #是否将每个样本数据除以自身的标准差
                 zca_whitening = False,                  #是否对输入数据施以ZCA白化
                 rotation_range = 20,                    #数据提升时图片随机转动的角度(范围为0～180)
                 width_shift_range  = 0.2,               #数据提升时图片水平偏移的幅度（单位为图片宽度的占比，0~1之间的浮点数）
                 height_shift_range = 0.2,               #同上，只不过这里是垂直
                 horizontal_flip = True,                 #是否进行随机水平翻转
                 vertical_flip = False)                  #是否进行随机垂直翻转

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

        if resized_im.shape[0] != 224 or resized_im.shape[1] != 224:
            print("invalid shape")

        # cv2.imwrite(image_name, resized_im)
    else:
        print(image_name+" error " + str(len(faces)))
    return resized_im


def randomUpdate(img):

    img = scipy.misc.toimage(img)

    # 旋转
    rotate = random.random() * 30 - 30
    image_rotated = img.rotate(rotate)

    # 亮度
    enh_bri = ImageEnhance.Brightness(image_rotated)
    bright = random.random() * 0.8 + 0.6
    image_brightened = enh_bri.enhance(bright)
    # image_brightened.show()

    # 对比度
    enh_con = ImageEnhance.Contrast(image_brightened)
    contrast = random.random() * 0.6 + 0.7
    image_contrasted = enh_con.enhance(contrast)
    # image_contrasted.show()

    # 色度
    enh_col = ImageEnhance.Color(image_contrasted)
    color = random.random() * 0.6 + 0.7
    image_colored = enh_col.enhance(color)

    enhance_im = np.asarray(image_colored)

    return enhance_im

lable_distribution = []

rating_files = ['female_white_images.csv',
                'female_yellow_images.csv',
                'male_white_images.csv',
                'male_yellow_images.csv',
                'remainder_images.csv']

pre_vote_image_name = ''
pre_vote_image_score1_cnt = 0
pre_vote_image_score2_cnt = 0
pre_vote_image_score3_cnt = 0
pre_vote_image_score4_cnt = 0
pre_vote_image_score5_cnt = 0

for rating_file_name in rating_files:

    rating_file = open(rating_path+rating_file_name, 'r')

    lines = rating_file.readlines();
    lines.pop(0)
    lineIdx = 0

    for line in lines:

        line = line.strip().split(',')
        lineIdx += 1;
        curr_row_image_name = line[1]
        score = int(line[2])

        if pre_vote_image_name == '':
            pre_vote_image_name = curr_row_image_name

        # 某个人的投票分数统计完成，计算标签分布
        if (curr_row_image_name != pre_vote_image_name) or (lineIdx == lines.__len__()):
            total_vote_cnt = pre_vote_image_score1_cnt + pre_vote_image_score2_cnt + pre_vote_image_score3_cnt + pre_vote_image_score4_cnt + pre_vote_image_score5_cnt
            score1_ld = pre_vote_image_score1_cnt / total_vote_cnt
            score2_ld = pre_vote_image_score2_cnt / total_vote_cnt
            score3_ld = pre_vote_image_score3_cnt / total_vote_cnt
            score4_ld = pre_vote_image_score4_cnt / total_vote_cnt
            score5_ld = pre_vote_image_score5_cnt / total_vote_cnt
            # print('投票对象：' + pre_vote_image_name)
            # print('1分标签分布:%1.2f ' % (score1_ld))
            # print('2分标签分布:%1.2f ' % (score2_ld))
            # print('3分标签分布:%1.2f ' % (score3_ld))
            # print('4分标签分布:%1.2f ' % (score4_ld))
            # print('5分标签分布:%1.2f ' % (score5_ld))

            im = detectFace(face_cascade, data_path, pre_vote_image_name)

            if isinstance(im, numpy.ndarray):
                # 原始数据样本
                normed_im = (im - 127.5) / 127.5

                ld = []
                ld.append(score1_ld)
                ld.append(score2_ld)
                ld.append(score3_ld)
                ld.append(score4_ld)
                ld.append(score5_ld)
                lable_distribution.append([pre_vote_image_name, normed_im, ld])

            else:
                print(pre_vote_image_name + " 未检测到人脸，丢弃样本")

            pre_vote_image_name = curr_row_image_name
            pre_vote_image_score1_cnt = 0
            pre_vote_image_score2_cnt = 0
            pre_vote_image_score3_cnt = 0
            pre_vote_image_score4_cnt = 0
            pre_vote_image_score5_cnt = 0

        # 统计投票分数，计算标签分布
        if score == 1:
            pre_vote_image_score1_cnt += 1
        elif score == 2:
            pre_vote_image_score2_cnt += 1
        elif score == 3:
            pre_vote_image_score3_cnt += 1
        elif score == 4:
            pre_vote_image_score4_cnt += 1
        elif score ==5:
            pre_vote_image_score5_cnt += 1

    rating_file.close()


data_split_index = int(lable_distribution.__len__() - lable_distribution.__len__()*0.1)

random.shuffle(lable_distribution)
test_lable_distribution = lable_distribution[data_split_index:]
train_lable_distribution = lable_distribution[:data_split_index]


train_data_len = train_lable_distribution.__len__()
for i in range(0, train_data_len):
    # 增强数据样本
    im = train_lable_distribution[i][1]
    enhance_im = randomUpdate(im)
    enhance_normed_im = (enhance_im - 127.5) / 127.5

    train_lable_distribution.append([pre_vote_image_name, enhance_normed_im, ld])

    # im_newshape = im.reshape((1,) + im.shape)
    # i = 0
    # for batch in datagen.flow(im_newshape,
    #                           batch_size=1):  # ,
    #     # save_to_dir=data_path+'aug',
    #     # save_prefix='aug',
    #     # save_format='jpg'):
    #
    #     batch = batch.reshape(im.shape)
    #     normed_batch = (batch - 127.5) / 127.5
    #
    #     train_image_data.append(normed_batch)
    #
    #     train_lable_distribution.append([pre_vote_image_name, ld])
    #
    #     i += 1
    #     if i > 3:
    #         break  # otherwise the generator would loop indefinitely

# pickle.dump(train_image_data, open('train_image_data.dat','wb'))
random.shuffle(train_lable_distribution)
pickle.dump(train_lable_distribution, open('train_lable_distribution.dat','wb'))

# pickle.dump(test_image_data, open('test_image_data.dat','wb'))
random.shuffle(test_lable_distribution)
pickle.dump(test_lable_distribution, open('test_lable_distribution.dat','wb'))
