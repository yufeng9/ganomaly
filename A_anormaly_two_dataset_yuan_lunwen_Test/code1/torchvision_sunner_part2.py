# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:49:36 2020

@author: zh
"""

from torchvision import transforms as torchvision_transforms
from code1.out import INFO, DEPRECATE
from code1.constant import *
import code1.setting as setting

from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import PIL

from skimage import transform

from code1.base import OP

from code1.constant import BCHW2BHWC

class ToTensor():
    def __init__(self):
        """
            Change the tensor into torch.Tensor type
            However, if the input is PIL image, then the original ToTensor will be used

            For the range of output tensor:
                1. [0~255] => [0~1] if the image is PIL object
                2. otherwise the value range doesn't change
                将张量更改为Torch.Tensor类型
             但是，如果输入为PIL图像，则将使用原始的ToTensor

             对于输出张量的范围：
                 1. [0〜255] => [0〜1]，如果图像是PIL对象
                 2.否则数值范围不会改变
        """
        INFO("Applied << %15s >>" % self.__class__.__name__)
        self.official_op_obj = transforms.ToTensor()

    def __call__(self, tensor):
        """
            Arg:    tensor - The torch.Tensor or other type. The tensor you want to deal with
        """
        if isinstance(tensor, PIL.Image.Image):
            # If the tensor is PIL image, then we use official torchvision ToTensor to deal with
            tensor = self.official_op_obj(tensor)

        elif isinstance(tensor, list):
            # If the tensor is list of PIL image, then we use official torchvision ToTensor iteratively to deal with
            tensor = torch.stack([self.official_op_obj(t) for t in tensor], 0)
        
        elif isinstance(tensor, np.ndarray):
            # Or we only transfer as TorchTensor
            tensor = torch.from_numpy(tensor)

        return tensor

class ToFloat():
    def __init__(self):
        """
            Change the tensor into torch.FloatTensor
        """        
        INFO("Applied << %15s >>" % self.__class__.__name__)

    def __call__(self, tensor):
        """
            Arg:    tensor - The torch.Tensor object. The tensor you want to deal with
        """
        tensor = tensor.float()
        return tensor
class Transpose():
    def __init__(self, direction = BHWC2BCHW):
        """
            Transfer the rank of tensor into target one

            Arg:    direction   - The direction you want to do the transpose
        """        
        self.direction = direction
        if self.direction == BHWC2BCHW:
            INFO("Applied << %15s >>, The rank format is BCHW" % self.__class__.__name__)
        elif self.direction == BCHW2BHWC:
            INFO("Applied << %15s >>, The rank format is BHWC" % self.__class__.__name__)
        else:
            raise Exception("Unknown direction symbol: {}".format(self.direction))

    def __call__(self, tensor):
        """
            Arg:    tensor - The torch.Tensor object. The tensor you want to deal with
        """
        if self.direction == BHWC2BCHW:
            tensor = tensor.transpose(-1, -2).transpose(-2, -3)
        else:
            tensor = tensor.transpose(-3, -2).transpose(-2, -1)
        return tensor

class RandomHorizontalFlip():
    def __init__(self, p = 0.5):
        """
            Flip the tensor toward horizontal direction randomly

            Arg:    p   - The random probability to filp the tensor
        """
        INFO("Applied << %15s >>" % self.__class__.__name__)
        INFO("* Notice: the rank format of input tensor should be 'BCHW'")
        if p < 0.0 or p > 1.0:
            raise Exception("The parameter 'p' should in (0, 1], but get {}".format(p))
        self.p = p

    def __call__(self, tensor):
        """
            Arg:    tensor - The torch.Tensor object. The tensor you want to deal with
        """
        if setting.random_seed < self.p:
            dim_idx = len(tensor.size()) - 1
            tensor_list = list(torch.split(tensor, 1, dim=dim_idx))
            tensor_list = list(reversed(tensor_list))
            tensor = torch.cat(tensor_list, dim_idx)
        return tensor

class RandomVerticalFlip():
    def __init__(self, p = 0.5):
        """
            Flip the tensor toward vertical direction randomly

            Arg:    p   - The random probability to filp the tensor
        """
        INFO("Applied << %15s >>" % self.__class__.__name__)
        INFO("* Notice: the rank format of input tensor should be 'BCHW'")
        if p < 0.0 or p > 1.0:
            raise Exception("The parameter 'p' should in (0, 1], but get {}".format(p))
        self.p = p

    def __call__(self, tensor):
        """
            Arg:    tensor - The torch.Tensor object. The tensor you want to deal with
        """
        if setting.random_seed < self.p:
            dim_idx = len(tensor.size()) - 2
            tensor_list = list(torch.split(tensor, 1, dim=dim_idx))
            tensor_list = list(reversed(tensor_list))
            tensor = torch.cat(tensor_list, dim_idx)
        return tensor

class GrayStack():
    def __init__(self, direction = BHW2BHWC):
        """
            Stack the gray-scale image for 3 times to become RGB image
            If the input is already RGB image, this function do nothing

            Arg:    direction   - The stack direction you want to conduct
        """
        INFO("Applied << %15s >>" % self.__class__.__name__)
        INFO("* Notice: the rank format of input tensor should be 'BHWC'")
        self.direction = direction

    def __call__(self, tensor):
        """
            Arg:    tensor - The torch.Tensor object. The tensor you want to deal with
        """
        if isinstance(tensor, np.ndarray):
            tensor = tensor.from_numpy(tensor)
            back_to_numpy = True
        else:
            back_to_numpy = False
        if self.direction == len(tensor.size()):
            if tensor.size(-1) == 1:
                tensor = torch.cat([tensor, tensor, tensor], -1)
        elif self.direction == (len(tensor.size()) + 1):
            tensor = torch.stack([tensor, tensor, tensor], -1)
        if back_to_numpy:
            tensor = tensor.cpu().numpy()
        return tensor
"""
class resize是用于重新约定图片长和宽的对象
"""
class Resize():
    def __init__(self, output_size):
        """
            Rescale the tensor to the desired size
            This function only support for nearest-neighbor interpolation
            Since this mechanism can also deal with categorical data

            Arg:    output_size - The tuple (H, W)
            将张量调整为所需大小
             该功能仅支持最近邻插值
             由于此机制还可以处理分类数据

             Arg：output_size-元组（H，W）
        """
        self.output_size = output_size
        self.op = torchvision_transforms.Resize(output_size, interpolation = Image.NEAREST)
        INFO("Applied << %15s >>" % self.__class__.__name__)
        INFO("* Notice: the rank format of input tensor should be 'BCHW'")

    def __call__(self, tensor):
        if isinstance(tensor, Image.Image):
            tensor = self.op(tensor)
        elif isinstance(tensor, list):
            tensor = [self.op(t) for t in tensor]
        return tensor



class Normalize(OP):
    def __init__(self, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
        """
            Normalize the tensor with given mean and standard deviation
            We recommand you to set mean as [0.5, 0.5, 0.5], and std as [0.5, 0.5, 0.5]
            Then the range will locate in [-1, 1]
            * Notice: If you didn't give mean and std, then we will follow the preprocessing of VGG
                      However, The range is NOT located in [-1, 1]

            Args:
                mean        - The mean of the result tensor
                std         - The standard deviation
            用给定的均值和标准差对张量进行归一化
             我们建议您将平均值设置为[0.5，0.5，0.5]，std设置为[0.5，0.5，0.5]
             然后范围将位于[-1，1]
             *注意：如果您没有给出均值和标准差，那么我们将遵循VGG的预处理
                       但是，范围不在[-1，1]中

            ARGS：
                 平均值-结果张量的平均值
                 std-标准偏差
        """
        self.mean = mean
        self.std  = std
        INFO("Applied << %15s >>" % self.__class__.__name__)
        INFO("* Notice: the rank format of input tensor should be 'BCHW'")
        INFO("*****************************************************************")
        INFO("* Notice: You should must call 'ToFloat' before normalization")
        INFO("*****************************************************************")
        if self.mean == [0.485, 0.456, 0.406] and self.std == [0.229, 0.224, 0.225]:
            INFO("* Notice: The result will NOT locate in [-1, 1]")

    def work(self, tensor):
        """
            Normalize the tensor

            Arg:    tensor  - The np.ndarray object. The tensor you want to deal with
            Ret:    The normalized tensor
        """
        if tensor.shape[0] != len(self.mean):
            raise Exception("The channel size should be {}, but the shape is {}".format(len(self.mean), tensor.shape))
        
        # Record the minimun and maximun value (in order to check if the function work normally)
        min_v, max_v = np.min(tensor), np.max(tensor)

        # Normalize with the given mean and std
        result = []
        for t, m, s in zip(tensor, self.mean, self.std):
            result.append((t - m) / s)
        tensor = np.asarray(result)

        # Check if the normalization can really work
        if self.mean != [1.0, 1.0, 1.0] and self.std != [1.0, 1.0, 1.0]:
            if np.min(tensor) == min_v and np.max(tensor) == max_v:
                raise Exception("Normalize can only work with float tensor",
                    "Try to call 'ToFloat()' before normalization")
        return tensor
    




channel_op = None       # Define the channel op which will be used in 'asImg' function

def asImg(tensor, size = None):
    """
        This function provides fast approach to transfer the image into numpy.ndarray
        This function only accept the output from sigmoid layer or hyperbolic tangent output

        Arg:    tensor  - The torch.Variable object, the rank format is BCHW or BHW
                size    - The tuple object, and the format is (height, width)
        Ret:    The numpy image, the rank format is BHWC
    """
    global channel_op
    result = tensor.detach()

    # 1. Judge the rank first
    if len(tensor.size()) == 3:
        result = torch.stack([result, result, result], 1)

    # 2. Judge the range of tensor (sigmoid output or hyperbolic tangent output)
    min_v = torch.min(result).cpu().data.numpy()
    max_v = torch.max(result).cpu().data.numpy()
    if max_v > 1.0 or min_v < -1.0:
        raise Exception('tensor value out of range...\t range is [' + str(min_v) + ' ~ ' + str(max_v))
    if min_v < 0:
        result = (result + 1) / 2

    # 3. Define the BCHW -> BHWC operation
    if channel_op is None:
        channel_op = Transpose(BCHW2BHWC)

    # 3. Rest               
    result = channel_op(result)
    result = result.cpu().data.numpy()
    if size is not None:
        result_list = []
        for img in result:
            result_list.append(transform.resize(img, (size[0], size[1]), mode='constant', order=0) * 255)
        result = np.stack(result_list, axis=0)
    else:
        result *= 255.
    result = result.astype(np.uint8)
    return result