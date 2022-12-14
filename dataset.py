import os
import torch
import numpy as np
import torch.utils.data as data

from tools.utils import add_classification_feat


class VideoFeatDataset(data.Dataset):
    def __init__(self, vpath, apath, cpath=None, supervised=False):
        self.apath = apath
        self.vpath = vpath
        self.classpath = cpath
        self.supervised = supervised

    def __getitem__(self, index):
        vfeat = np.load(os.path.join(self.vpath, '%04d.npy'%(index))).astype('float32')
        afeat = np.load(os.path.join(self.apath, '%04d.npy'%(index))).astype('float32')

        if self.classpath is not None and self.supervised == False:
            class_labels = np.load(self.classpath)
            class_flag = class_labels[index] + 1     # in case that index is 0
            vfeat = add_classification_feat(class_flag, vfeat, amp=1)
            afeat = add_classification_feat(class_flag, afeat, amp=225)
        elif self.classpath is not None and self.supervised == True:
            class_labels = np.load(self.classpath)
            class_labels = torch.tensor(class_labels).float().cuda()
            return vfeat, afeat, class_labels[index]

        return vfeat, afeat

    def __len__(self):
        return len(os.listdir(self.apath))

    def loader(self, filepath):
        return np.load(filepath)
