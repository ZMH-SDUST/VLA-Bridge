# -*- coding: utf-8 -*-
"""
@Time ： 2024/10/22 21:47
@Auther ： Zzou
@File ：batch_draw_attention.py
@IDE ：PyCharm
@Motto ：ABC(Always Be Coding)
@Info ：
"""

import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pocket
import pocket.advis
import torch
from PIL import Image
from torchvision.datasets import Places365, CIFAR100
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
import clip
from visualization.visual import vis_grid_attention

model, preprocess = clip.load("ViT-B/32", jit=False)
model.cuda()

image_dir = ""
save_dir = ""

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

for file in os.listdir(image_dir):
    file_path = os.path.join(image_dir, file)
    image = Image.open(file_path)
    image_input = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        image_attention = model.encode_image_attention(image_input.cuda()).cpu()[0].reshape(7, 7)
    image_attention = image_attention.float()
    vis_grid_attention(
        img_path=file_path,
        save_dir=save_dir,
        attention_map=np.array(image_attention))
