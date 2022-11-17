import os
import torch
import numpy as np
from abc import abstractmethod


class AudioExtractor(torch.nn.Module):
    def __init__(self, cfg):
        super(AudioExtractor, self).__init__()
        self.model = torch.nn.Module()
        self.name = cfg.model

    @abstractmethod
    def forward(self, x):
        pass

    def extract_dir(self, dirname):
        os.system(f'mkdir {dirname}/{self.name}')
        vnames = os.listdir(os.path.join(dirname, 'audio'))
        for vname in vnames:
            sname = vname[:-4] + '.npy'
            feat = self.model.forward(os.path.join(dirname, 'audio', vname))
            print(feat.shape, sname)
            np.save(os.path.join(dirname, {self.name}, sname), feat.detach().cpu().numpy())