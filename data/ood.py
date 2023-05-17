import numpy as np
import sys
import os
import pickle
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
import sklearn.metrics as sk

# go through rigamaroo to do ...utils.display_results import show_performance

# mean and standard deviation of channels of CIFAR-10 images
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]

test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])


# /////////////// Detection Prelims ///////////////

batch_size = 128
concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out

def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])

def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr

def get_ood_scores(loader, net , device , ood_num_examples,in_dist=False,score = 'MSP'):
    _score = []
    _right_score = []
    _wrong_score = []
    flagg = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // batch_size and in_dist is False:
                break

            data = data.to(device)

            output = net(data)
            smax = to_np(F.softmax(output, dim=1))

            if score == 'energy':
                _score.append(-to_np((0.5 * torch.logsumexp(output / 0.5, dim=1))))
            elif score == 'MSP':  # original MSP and Mahalanobis (but Mahalanobis won't need this returned)
                    _score.append(-np.max(smax, axis=1))
            else:
                _score.append(to_np((output.size(1) / (output.size(1) + torch.sum(torch.exp(output), dim=1)))))
            if in_dist:
                preds = np.argmax(smax, axis=1)
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)

                _right_score.append(-np.max(smax[right_indices], axis=1))
                _wrong_score.append(-np.max(smax[wrong_indices], axis=1))
    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()


auroc_list, aupr_list, fpr_list = [], [], []
def get_and_print_results(ood_loader , net , device , traindata , in_score , ood_num_examples):
    aurocs, auprs, fprs = [], [], []


    out_score = get_ood_scores(ood_loader, net , device , ood_num_examples)

    measures = get_measures(-in_score, -out_score)
    aurocs.append(measures[0]);
    auprs.append(measures[1]);
    fprs.append(measures[2])

    # print(in_score[:3], out_score[:3])
    auroc = np.mean(aurocs);
    aupr = np.mean(auprs);
    fpr = np.mean(fprs)
    auroc_list.append(auroc);
    aupr_list.append(aupr);
    fpr_list.append(fpr)

    print('  FPR{:d} AUROC AUPR'.format(int(100*0.95)))
    print('& {:.2f} & {:.2f} & {:.2f}'.format(100*fpr, 100*auroc, 100*aupr))

# /////////////// End Detection Prelims ///////////////
def OOD_detection(net , traindata , device , dataroot):

    if traindata == 'CIFAR10':
        test_data = dset.CIFAR10(dataroot + '/cifar-10/', train=False, transform=test_transform)
        num_classes = 10
    else:
        test_data = dset.CIFAR100(dataroot + '/cifar-100/', train=False, transform=test_transform)
        num_classes = 100


    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False,num_workers=0, pin_memory=True)
    ood_num_examples = len(test_data)

    in_score, right_score, wrong_score = get_ood_scores(test_loader, net , device , ood_num_examples,in_dist=True)
    print('Using CIFAR-10 as typical data') if num_classes == 10 else print('Using CIFAR-100 as typical data')



    # /////////////// SVHN ///////////////
    ood_data = torchvision.datasets.SVHN(root=dataroot + '/svhn/', split="test",transform=trn.Compose([  trn.ToTensor(), trn.Normalize(mean, std)]), download=False)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=128, shuffle=False,num_workers=0, pin_memory=True)
    ood_num_examples = len(ood_data)

    print('\nSVHN Detection')
    get_and_print_results(ood_loader , net , device , traindata , in_score , ood_num_examples)

    # /////////////// LSUN-C ///////////////
    ood_data = dset.ImageFolder(root=dataroot + "/LSUN_C/",transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=128, shuffle=True,num_workers=0, pin_memory=True)
    print('\nLSUN_C Detection')
    get_and_print_results(ood_loader , net , device , traindata , in_score , ood_num_examples)

    # /////////////// iSUN ///////////////
    ood_data = dset.ImageFolder(root=dataroot+"/iSUN/",transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=128, shuffle=True,num_workers=0, pin_memory=True)
    print('\niSUN Detection')
    get_and_print_results(ood_loader , net , device , traindata , in_score , ood_num_examples)


