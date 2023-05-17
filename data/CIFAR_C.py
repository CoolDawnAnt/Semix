import numpy as np
import os
import PIL
import torch
import torchvision

from PIL import Image
from torch.utils.data import Subset
from torchvision import datasets
from torch.utils.data import DataLoader

def load_txt(path :str) -> list:
    return [line.rstrip('\n') for line in open(path)]

CORRUPTIONS = load_txt('./data/corruptions.txt')


class CIFAR_C(datasets.VisionDataset):
    def __init__(self, root: str, name: str,
                 transform=None, target_transform=None):
        assert name in CORRUPTIONS
        super(CIFAR_C, self).__init__(
            root, transform=transform,
            target_transform=target_transform
        )
        data_path = os.path.join(root, name + '.npy')
        target_path = os.path.join(root, 'labels.npy')

        self.data = np.load(data_path)
        self.targets = np.load(target_path)

    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return img, targets

    def __len__(self):
        return len(self.data)


def eval_CIFARC(model , traindata , data_root , transform , device):

    acc = 0
    tot = 0
    for ci, cname in enumerate(CORRUPTIONS):
            # print(cname)
            # load dataset
            if cname == 'natural':
                if (traindata == 'CIFAR10'):
                    dataset = datasets.CIFAR10(
                        os.path.join(data_root, 'cifar-10'),
                        train=False, transform = transform, download=False,
                    )
                else:
                    dataset = datasets.CIFAR100(
                        os.path.join(data_root, 'cifar-100'),
                        train=False, transform=transform, download=False,
                    )
            else:
                dataset = CIFAR_C(
                    os.path.join(data_root, 'cifar10-c' if traindata == 'CIFAR10' else 'cifar100-c'),
                    cname, transform=transform
                )
            loader = DataLoader(dataset, batch_size=128,shuffle=False, num_workers=4)

            correct = 0.
            total = 0.
            with torch.no_grad():
                for itr, (x, y) in enumerate(loader):
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, dtype=torch.int64, non_blocking=True)

                    z = model(x)

                    predict_y = torch.max(z, dim=1)[1]
                    correct += torch.eq(predict_y, y).cpu().sum().item()
                    total += y.size(0)

            acc += correct
            tot += 1

    acc = acc / tot * 100

    return acc