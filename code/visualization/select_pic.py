# -*- coding: utf-8 -*-
"""
@Time ： 2024/10/29 16:34
@Auther ： Zzou
@File ：select_pic.py
@IDE ：PyCharm
@Motto ：ABC(Always Be Coding)
@Info ：
"""
import os
import shutil

gt_folder = "../visualization_gt/hicodet"
raw_folder = "../visualization_pred_raw/zs/rare_first/hicodet"
our_folder = "../visualization_pred_ours/zs/rare_first/hicodet"
attn_folder = "../visualization_attn/"

gt_dict = {}
raw_dict = {}
our_dict = {}

for class_name in os.listdir(gt_folder):
    gt_class_files = os.listdir(os.path.join(gt_folder, class_name))
    for image_name in gt_class_files:
        image_name = image_name.split("_")[2]
        if image_name not in gt_dict.keys():
            gt_dict[image_name] = []
        gt_dict[image_name].append(class_name)

for class_name in os.listdir(raw_folder):
    raw_class_files = os.listdir(os.path.join(raw_folder, class_name))
    for image_name in raw_class_files:
        image_name = image_name.split("_")[2]
        if image_name not in raw_dict.keys():
            raw_dict[image_name] = []
        raw_dict[image_name].append(class_name)

for class_name in os.listdir(our_folder):
    our_class_files = os.listdir(os.path.join(our_folder, class_name))
    for image_name in our_class_files:
        image_name = image_name.split("_")[2]
        if image_name not in our_dict.keys():
            our_dict[image_name] = []
        our_dict[image_name].append(class_name)

attn_image_list = []
for image in gt_dict.keys():
    if image in our_dict.keys() and image in raw_dict.keys():
        attn_image ="HICO_test2015_" + image + ".jpg"
        flag = False
        for item in gt_dict[image]:
            if item in our_dict[image] and item not in raw_dict[image]:
                flag = True
                print("missing prediction of raw is: ", item)
        if flag:
            print(image)
            print(gt_dict[image])
            print(our_dict[image])
            print(raw_dict[image])
            attn_image_list.append(attn_image)
        add_wrong = difference = [item for item in raw_dict[image] if item not in gt_dict[image]]
        print("addition wrong prediction of raw is :", add_wrong)
        print("---------------------------------------------------------------------------------------")
