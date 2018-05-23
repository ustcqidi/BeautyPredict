import os
import random
from PIL import Image
from PIL import ImageEnhance

parent_path = os.path.dirname(os.getcwd())
parent_path = os.path.dirname(parent_path)
img_path = parent_path + "/dataset/SCUT-FBP5500/Images/"
save_path = parent_path + "/dataset/SCUT-FBP5500/enhanced_images/"


# 亮度
def randomUpdate(path, name):
    img = Image.open(path + name)
    # img.show()

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
    # image_colored.show()

    image_colored.save(save_path + name)


pathDir = os.listdir(img_path)
for file in pathDir:
    if file.endswith(".jpg"):
        randomUpdate(img_path, file)
