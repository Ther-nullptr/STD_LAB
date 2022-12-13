import os
import time

import numpy as np
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n # val * n: how many samples predicted correctly among the n samples
        self.count += n     # totoal samples has been through
        self.avg = self.sum / self.count

def save_checkpoint(model, output_path):
    torch.save(model, output_path)
    print("Checkpoint saved to {}".format(output_path))

def add_classification_feat(class_flag, feat, amp=1, bias=0):
    """
        position encoding for different classes, here it refers to 27.
        after encoding, the class label is changed into 128.
    """
    # base_number = 1000
    feat_dim = feat.shape[1]
    # expo = class_flag / (base_number ** (np.arange(feat_dim) / base_number))
    # internal = int(np.floor(feat_dim / 2))

    # encoding method : one hot
    class_feat = np.zeros((1, feat_dim), dtype=np.float32)
    class_feat[0][class_flag] =  amp + bias

    # encoding method : position encoding
    # class_feat = np.concatenate((
    #     amp * (np.sin(expo[:internal], dtype=np.float32)) + bias,
    #     amp * (np.cos(expo[internal:], dtype=np.float32)) + bias
    # ), axis=0)
    # class_feat = class_feat.reshape((1, feat_dim))
    return np.concatenate((feat, class_feat), axis=0)