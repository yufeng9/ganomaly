from code1.parse_part1 import parse_args

from code1.model import GANomaly2D
import code1.torchvision_sunner_part2 as sunnerTransforms
import torchvision as tv
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch
import cv2
import os
import numpy as np
from code1.difference import *
from collections import defaultdict
import json
import zipfile


def compre(file, zip_name):
    zp = zipfile.ZipFile(zip_name, 'a', zipfile.ZIP_DEFLATED)
    zp.write(file)
    zp.close()


def demo_part1():
    '''以下为修改图片加载部分'''
    args = parse_args(phase='demo')
    transforms = tv.transforms.Compose([
        sunnerTransforms.Resize(output_size=(args.H, args.W)),
        sunnerTransforms.ToTensor(),
        sunnerTransforms.ToFloat(),
        sunnerTransforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = tv.datasets.ImageFolder(args.demo, transform=transforms)
    loader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=2,
                        drop_last=True)

    '''数据加载修改结束'''

    # Create the model
    model = GANomaly2D(r=args.r, device=args.device)
    model.IO(args.resume, direction='load')

    # Demo!
    bar = tqdm(loader)
    model.eval()
    with torch.no_grad():
        for i, (img, _) in enumerate(bar):
            imgname = dataset.imgs[i][0]
            name_list = imgname.split('/')
            imgname = name_list[-1]
            z = model.forward(img)
            img, img_ = model.getImg()
            img = sunnerTransforms.asImg(img)[0]
            img_ = sunnerTransforms.asImg(img_)[0]
            residual_img, threshold_img = difference(img, img_, 100)  # 測試圖片 #生成圖片 #門檻值
            result = np.hstack((img, img_, threshold_img, residual_img))
            img_path = os.path.join('../temp_data/', imgname)
            cv2.imwrite(img_path, threshold_img)




