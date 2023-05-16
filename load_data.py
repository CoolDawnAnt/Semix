from torchvision import datasets
from torchvision import transforms
import torch
import torchvision
from torch.utils.data import DataLoader

def getdata(dataset , num_workers , batch_size):
    if (dataset == 'cifar10') or (dataset == 'cifar100'):
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif (dataset == 'svhn'):
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif (dataset == 'tinyimagenet'):
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    if (dataset == 'svhn'):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    elif (dataset == 'tinyimagenet'):
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    stride = 1
    if (dataset == 'cifar10'):
        train_dataset = torchvision.datasets.CIFAR10(root='./dataset/cifar10/', train=True, transform=transform_train,download=False)
        test_dataset = datasets.CIFAR10(root='./dataset/cifar10/', train=False, download=False, transform=transform_test)
        num_classes = 10
    elif (dataset == 'cifar100'):
        train_dataset = torchvision.datasets.CIFAR100(root='./dataset/cifar100/', train=True, transform=transform_train,download=False)
        test_dataset = datasets.CIFAR100(root='./dataset/cifar100/', train=False, download=False,transform=transform_test)
        num_classes = 100
    elif (dataset == 'svhn'):
        train_dataset = torchvision.datasets.SVHN(root='./dataset/svhn/', split='train', transform=transform_train,download=False)
        test_dataset = datasets.SVHN(root='./dataset/svhn/', split='test', download=False, transform=transform_test)
        num_classes = 10
    elif (dataset == 'tinyimagenet'):
        train_dataset = datasets.ImageFolder(root='./dataset/tiny-imagenet-200/train', transform=transform_train)
        test_dataset = datasets.ImageFolder(root='./dataset/tiny-imagenet-200/val/images', transform=transform_test)
        num_classes = 200
        stride = 2

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    return  train_loader , test_loader , num_classes , stride