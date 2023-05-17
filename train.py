import numpy as np
import torchvision
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random
import os
from load_data import getdata
import pickle
from copy import deepcopy
import models
import argparse
import util

parser = argparse.ArgumentParser(description='',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'svhn', 'tinyimagenet'])
parser.add_argument('--arch', type=str, default='PreResNet18', choices=['PreResNet18', 'wrn28_10', 'VGG19' , 'LeNet'])
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--zeta', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--lr_decay', type=float, default=0.0001)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--base', type=str, default='Mixup' , choices=['Mixup' , 'Cutmix'] , help='Choose a basement Mixup method like original Mixup or Cutmix')

parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--ES',type=bool,default=False)
parser.add_argument('--ES_epoch',type=int,default=0)
parser.add_argument('--resume', '-r', action='store_true',help='resume from checkpoint')

best_acc = 0.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
loss_func = nn.CrossEntropyLoss()
loss_func_reg = nn.MSELoss()


opt = parser.parse_args()
if (opt.base == 'Mixup'):
    mixdata = util.mixup
else:
    mixdata = util.cutmix


random.seed(opt.seed)
os.environ['PYTHONHASHSEED'] = str(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)


train_loader , test_loader , num_classes , stride = getdata(opt.dataset , opt.num_workers , opt.batch_size)

start_epoch = 0
net = models.__dict__[opt.arch](num_classes, stride)
optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=opt.lr_decay)
if opt.resume:
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'

    model_save_name = opt.dataset + '_' + opt.arch + '_alpha' + str(opt.alpha) + '_zeta' + str(opt.zeta) + '_seed' + str(opt.seed) + '.pth'
    checkpoint = torch.load('./checkpoint/' + model_save_name)
    net.load_state_dict(checkpoint['state_dict'])
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch'] + 1
    optimizer.load_state_dict(checkpoint['optimizer'])
    opt = checkpoint['opt']
net.to(device)
cudnn.benchmark = True
lr = opt.lr



def checkpoint(epoch):
    state = {
        'setting': opt,
        'epoch': epoch,
        'state_dict': net.state_dict(),
        'best_acc': best_acc,
        'optimizer': optimizer.state_dict(),
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    model_save_name = opt.dataset + '_' + opt.arch + '_alpha' + str(opt.alpha) + '_zeta' + str(opt.zeta) + '_seed' + str(opt.seed) + '.pth'
    torch.save(state, './checkpoint/' + model_save_name)


def test(epoch):
    global best_acc
    net.eval()
    total = 0
    correct = 0.
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader, 0):
            testx, testy = data[0].to(device), data[1].to(device)

            outputs = net(testx)
            predict_y = torch.max(outputs, dim=1)[1]
            correct += torch.eq(predict_y, testy).cpu().sum().item()
            total += testx.size()[0]
            util.progress_bar(batch_idx, len(test_loader),
                         'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (0 / (batch_idx + 1), 100. * correct / total,
                            correct, total))
    accuracy = correct / total
    if (accuracy > best_acc):
        best_acc = accuracy
        checkpoint(epoch)


    print('total test accuracy %f%%' %(100 * accuracy) )

    return accuracy


def adjust_learning_rate(epoch, optimizer):
    global lr
    if (epoch == 100) or (epoch == 150):
        lr /= 10
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def train(epoch):
    adjust_learning_rate(epoch, optimizer)
    net.train()
    for batch_idx, data in enumerate(train_loader, 0):

        inputs, labels = data[0], data[1]

        if (opt.ES == True) and (epoch > opt.ES_epoch):
            inputs = inputs.to(device)
            labels = labels.to(device)
            loss = loss_func(outputs, labels)
        else:
            inputs_a, inputs_b, mixed_inputs, labels_a, labels_b, lam = mixdata(inputs, labels, opt.alpha)

            inputs_a = inputs_a.to(device)
            inputs_b = inputs_b.to(device)
            mixed_inputs = mixed_inputs.to(device)

            labels_a = labels_a.to(device)
            labels_b = labels_b.to(device)

            outputs_a = net(inputs_a)
            feature_a = net.feature

            outputs_b = net(inputs_b)
            feature_b = net.feature

            outputs = net(lam * inputs_a + (1 - lam) * inputs_b)
            feature = net.feature


            loss = lam * loss_func(outputs, labels_a) + (1 - lam) * loss_func(outputs, labels_b) \
                   + opt.zeta * loss_func_reg(feature, lam * feature_a + (1 - lam) * feature_b)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return


for epoch in range(start_epoch , opt.epoch):
    print("Epoch %d" % (epoch))
    train(epoch)
    test_acc = test(epoch)

