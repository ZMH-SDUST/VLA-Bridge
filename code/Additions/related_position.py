# -*- coding: utf-8 -*-
"""
@Time ： 2024/5/25 10:49
@Auther ： Zzou
@File ：related_position.py
@IDE ：PyCharm
@Motto ：ABC(Always Be Coding)
@Info ：
"""
import json
import math
import os.path
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.manifold import TSNE


def draw_bounding_box(image_path, x1, y1, x2, y2):
    image = cv2.imread(image_path)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("Image with Bounding Box", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def relative_features(human_box, object_box):
    human_center_x = (human_box[0] + human_box[2]) / 2
    human_center_y = (human_box[1] + human_box[3]) / 2
    object_center_x = (object_box[0] + object_box[2]) / 2
    object_center_y = (object_box[1] + object_box[3]) / 2
    relative_x = object_center_x - human_center_x
    relative_y = object_center_y - human_center_y
    ang = math.atan2(relative_y, relative_x)
    ang = (ang + math.pi) / (2 * math.pi)
    distance = (relative_x ** 2 + relative_y ** 2) ** 0.5
    related_distance = distance / (human_box[2] + human_box[1])
    return distance, related_distance, ang


def draw_scatter(verb_info):
    for i, key in enumerate(verb_info.keys()):
        category_features = verb_info[key]
        x = [feature[0] for feature in category_features]
        y = [feature[1] for feature in category_features]
        if i < 5:
            plt.scatter(x, y, label=f"Category {key}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Scatter Plot of Categories")
    plt.savefig("scatter.jpg")


def draw_tsne(verb_info, selected_keys=None):
    if selected_keys is None:
        selected_keys = [10, 15, 40, 47]
    features = []
    labels = []
    for key in verb_info.keys():
        if key in selected_keys:
            values = verb_info[key]
            features.extend(values)
            labels.extend([key] * len(values))
    features_array = np.array(features)
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(features_array)
    unique_labels = np.unique(labels)
    for label in unique_labels:
        category_tsne = features_tsne[labels == label]
        x = category_tsne[:, 0]
        y = category_tsne[:, 1]
        plt.scatter(x, y, s=10, label=f"Category {label}")
    plt.legend()
    plt.title("T-SNE Visualization of Verb Categories")
    plt.savefig("tsne.jpg")


def draw_hist(verb_info, verb_labels, selected_keys=None):
    if selected_keys is None:
        selected_keys = 1
    category_features = verb_info[selected_keys]
    data = [feature[1] for feature in category_features]
    plt.hist(data, bins=18, edgecolor='black')
    plt.title("Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig(
        os.path.join("Histogram", verb_labels[selected_keys - 1] + ".jpg"))