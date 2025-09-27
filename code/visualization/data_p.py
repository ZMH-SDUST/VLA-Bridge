# -*- coding: utf-8 -*-
"""
@Time ： 2024/10/22 9:08
@Auther ： Zzou
@File ：data.py
@IDE ：PyCharm
@Motto ：ABC(Always Be Coding)
@Info ：
"""

# --------------------------------------
# -*- coding: utf-8 -*-
# @Time : 2022/8/24 16:43
# @Author : wzy
# @File : data.py
# ---------------------------------------
import cv2
import numpy as np
from torchvision.transforms import transforms


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def data_load(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def data_process(img):
    img = np.array(img)  # Image->ndarray
    img = transform(img)  # tensor
    img = img.unsqueeze(0)
    print(f'image dimension:{img.size()}')  # [1, 3, 512, 512]
    return img


def data_to_model(conv_layers, img):
    output = [conv_layers[0](img)]
    for i in range(1, len(conv_layers)):
        output.append(conv_layers[i](output[-1]))
    return output


def data_to_tensor(img_path):
    img = data_load(img_path)
    img = transform(img)
    return img.unsqueeze(0)