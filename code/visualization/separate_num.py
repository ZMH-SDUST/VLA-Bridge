# -*- coding: utf-8 -*-
"""
@Time ： 2024/10/22 9:08
@Auther ： Zzou
@File ：separate_num.py
@IDE ：PyCharm
@Motto ：ABC(Always Be Coding)
@Info ：
"""

# --------------------------------------
# -*- coding: utf-8 -*-
# @Time : 2022/8/24 15:56
# @Author : wzy
# @File : separate_num.py
# @reference : https://blog.csdn.net/qq_36607894/article/details/103595912
# ---------------------------------------
import numpy as np


def separate(input):

    start = int(np.sqrt(input))
    factor = input / start
    while not is_integer(factor):
        start += 1
        factor = input / start
    return int(factor), start


def is_integer(number):
    if int(number) == number:
        return True
    else:
        return False


if __name__ == '__main__':
    print(separate(120))