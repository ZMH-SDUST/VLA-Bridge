# -*- coding: utf-8 -*-
"""
@Time ： 2024/10/22 9:05
@Auther ： Zzou
@File ：visual.py
@IDE ：PyCharm
@Motto ：ABC(Always Be Coding)
@Info ：
"""

# --------------------------------------
# -*- coding: utf-8 -*-
# @Time : 2022/8/24 15:39
# @Author : wzy
# @File : visual.py
# ---------------------------------------
import math
import os.path

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from data_p import data_load, transform, data_to_tensor
from separate_num import separate
import matplotlib
import seaborn as sns
from einops import rearrange

matplotlib.use('Agg')


def vis_filter(model_weights, layer):
    """
    :param model_weights: 传入整个模型的权重
    :param layer: 选择可视化哪一层
    :return:
    """
    print('p====================卷积核可视化====================q')
    filter_num = model_weights[layer].shape[0]
    filter_channel = model_weights[layer].shape[1]
    print(model_weights[layer].shape)
    print(f'该层一共有[{filter_num}]个卷积核,每个卷积核的维度为[{filter_channel}]')
    row, column = separate(filter_num)
    for j in range(filter_channel):
        plt.figure(figsize=(5, 5))
        for i, filter in enumerate(model_weights[layer]):
            plt.subplot(row, column, i + 1)
            plt.imshow(filter[j, :, :].detach(), cmap='gray')
            plt.axis('off')
        plt.savefig(f'./imgs_out/filter_layer{layer + 1}_channel{j}.png')
        plt.show()
    print('b==================卷积核可视化结束===================d')


def vis_image(image):
    h, w = image.shape[0], image.shape[1]
    plt.subplots(figsize=(w * 0.01, h * 0.01))
    plt.imshow(image, alpha=1)
    return h, w


def vis_feature(features, num_layers):
    """
    :param features: 特征图(即图像经过卷积后的样子)
    :param num_layers:所有的特征图层数
    :return:
    """
    print('p====================特征图可视化====================q')
    for num_layer in range(len(features)):
        plt.figure(figsize=(5, 5))
        layer_vis = features[num_layer][0, :, :, :]
        layer_vis = layer_vis.data
        print(f'[{num_layer + 1}] feature size :{layer_vis.size()}')
        feature_num = layer_vis.shape[0]
        row, column = separate(feature_num)

        for i, filter in enumerate(layer_vis):
            plt.subplot(row, column, i + 1)
            plt.imshow(filter)
            plt.axis('off')
        print(f'Saving layer feature maps : [{num_layer + 1}]/[{num_layers}] ')
        plt.savefig(f'./imgs_out/layer_{num_layer}.png')
        plt.close()
    print('b==================特征图可视化结束===================d')


def vis_attention_matrix(attention_map, index=0, cmap="YlGnBu"):
    """
    :param attention_map: 注意力得分矩阵
    :param index: map编号,便于多个注意力可视化的存储
    :param cmap: 颜色样式
    :return:
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        attention_map,
        vmin=0.0, vmax=1.0,
        cmap=cmap,
        # annot=True,
        square=True)
    plt.savefig(f'./imgs_out/attention_matrix_{index}.png')
    print(f'[attention_matrix_{index}.png] is generated')


def vis_grid_attention(img_path, save_dir, attention_map, cmap='jet'):
    """
    :param img_path:图像路径
    :param attention_map:注意力图
    :param cmap: cmap是图像的颜色类型，有很多预设的颜色类型
    :return:
    """
    # draw the img
    img = data_load(img_path)
    h, w = vis_image(img)

    # draw the attention
    map = cv2.resize(attention_map, (w, h))
    normed_map = map / map.max()
    normed_map = (normed_map * 255).astype('uint8')
    plt.imshow(normed_map, alpha=0.4, interpolation='nearest', cmap=cmap)

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    img_name = img_path.split('\\')[-1]
    plt.savefig(os.path.join(save_dir, img_name))
    plt.close()


def vis_img_patch(patch_tensor, patch_size, mask=None):
    """
    将图片划分成patch块并显示
    :param patch_tensor: (B,NUM,patch_size^2*C) 其中NUM是patch块的数量
    :param patch_size: patch的边长(默认正方形)
    :param mask:masking函数得到的mask矩阵，代表哪些patch被mask
    :return:
    """
    if mask is not None:
        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, 16 ** 2 * 3)
        patch_tensor = patch_tensor * (1 - mask)
    print(patch_tensor.shape)
    patch_tensor = patch_tensor.squeeze(0)
    n, d = patch_tensor.shape
    h = w = int(math.sqrt(n))
    plt.figure(figsize=(5, 5))
    img = rearrange(patch_tensor, '(h w) (n1 n2 c) -> (h n1) (w n2) c', h=h, w=w, n1=patch_size, n2=patch_size)
    plt.imshow(img)
    plt.axis('off')

    if mask is None:
        plt.savefig(f'./imgs_out/img_patch.png')
    else:
        plt.savefig(f'./imgs_out/masked_patch.png')
