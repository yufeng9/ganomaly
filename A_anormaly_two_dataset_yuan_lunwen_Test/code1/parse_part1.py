from code1.utils import INFO, showParameters
import argparse
import torch
import os




def parse_args(phase='train'):
    """
        Parse the argument for training procedure
        ------------------------------------------------------
                        [Argument explain]

        --train         : The folder path of training image (only normal)
        --demo          : The folder path of inference image (normal + abnormal)
        --resume        : The path of pre-trained model
        --det           : The path of destination model you want to store into
        --H             : The height of image
                          Default is 240
        --W             : The width of image
                          Default is 320
        --r             : The ratio of channel you want to reduce
                          Default is 1
        --batch_size    : The batch size in single batch
                          Default is 2
        --n_iter        : Total iteration
                          Default is 1 (30000 is recommand)
        --record_iter   : The period to record the render image and model parameters
                          Default is 1 (200 is recommand)

        ------------------------------------------------------
        Arg:    phase   (Str)   - The symbol of program (train or demo)
        Ret:    The argparse object
    """
    parser = argparse.ArgumentParser()
    if phase == 'train':
        parser.add_argument('--train', type=str, default='../raw_data/round_test/part1/')
    if phase == 'demo':
        parser.add_argument('--demo', type=str, default='../raw_data/round_test/part1/') # lunwen_yuan_without_data
    parser.add_argument('--resume', type=str, default='../model/flip_lunwen_yuan_w_con_10_w_enc_0_01_loss1.pkl')  # result300_0715 flip_fe_loss
    parser.add_argument('--det', type=str, default='../model/flip_lunwen_yuan_w_con_10_w_enc_0_01_loss1.pkl') # result_new_two_datasets_yuan_lunwen
    parser.add_argument('--H', type=int, default=1024)
    parser.add_argument('--W', type=int, default=1024)# flip_lunwen_yuan_w_con_30_w_enc_10
    parser.add_argument('--r', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_iter', type=int, default=100)
    parser.add_argument('--record_iter', type=int, default=1)

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    showParameters(vars(args))
    return args