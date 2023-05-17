import argparse
import glob
import numpy as np
import os
import pprint
import torch
import torchvision
import models

import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms
import data.CIFAR_C
import data.CIFAR10x
import data.ood

MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2023, 0.1994, 0.2010]






if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--arch',
        type=str, default='LeNet',
        help='model name'
    )
    parser.add_argument(
        '--dataset',
        type=str,default='CIFAR10',
        help='the dataset on which the model is trained',
    )
    parser.add_argument(
        '--weight_path',
        type=str, default='./checkpoint/cifar10_LeNet_alpha1_zeta0.5_seed0.pth',
        help='path to the dicrectory containing model weights',
    )

    parser.add_argument(
        '--data_root',
        type=str, default='C:/Users/Xtc/Desktop/res/dataset',
        help='root path to cifar10-c directory'
    )


    opt = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = models.__dict__[opt.arch](10 if opt.dataset == 'CIFAR10' else 100 , 1)
    checkpoint = torch.load(opt.weight_path)
    model.load_state_dict(checkpoint['state_dict'])

    model.to(device)
    model.eval()
    cudnn.benchmark = True
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    print('Corruption %f' % (data.CIFAR_C.eval_CIFARC(model , opt.dataset , opt.data_root , transform , device)))
    data.CIFAR10x.eval_CIFAR10x(model , opt.dataset , opt.data_root , transform , device)
    data.ood.OOD_detection(model, opt.dataset, device , opt.data_root)

