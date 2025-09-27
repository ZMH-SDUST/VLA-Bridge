# -*- coding: utf-8 -*-
"""
@Time ： 2024/10/22 8:22
@Auther ： Zzou
@File ：visualize_attention.py
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

parser = argparse.ArgumentParser()
parser.add_argument("--index", default=0, type=int, required=True)
parser.add_argument("--dataset", type=str, required=True)

model, preprocess = clip.load("ViT-B/32", jit=False)

args = parser.parse_args()

image = Image.open("")
image_input = preprocess(image).unsqueeze(0)

model.cuda()

with torch.no_grad():
    image_attention = model.encode_image_attention(image_input.cuda()).cpu()[0].reshape(7, 7)

image_attention = image_attention.float().unsqueeze(0)
attn_image = image.copy()  # PIL image
attn_save_filename = ""
pocket.advis.heatmap(image=attn_image, heatmaps=image_attention, ax=None, save_path=attn_save_filename)
plt.close()

