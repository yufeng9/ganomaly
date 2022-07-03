# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 17:53:28 2020

@author: zh
"""
from code1.demo_part1 import demo_part1

import os


def mkdir(path):
	folder = os.path.exists(path)
	if not folder:                   # 判断是否存在文件夹如果不存在则创建为文件夹

		os.makedirs(path)            # makedirs 创建文件时如果路径不存在会创建这个路径
	else:

		print("---  There is this folder!  ---")


if __name__ == '__main__':
    
    # mkdir('data/focusight1_round1_train_part1/TC_Images/')
    demo_part1()

    # os.rmdir('data')
    
