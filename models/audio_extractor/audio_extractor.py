import os
import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm
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
        os.system(f'mkdir {dirname}/afeat/{self.name}')
        vnames = os.listdir(os.path.join(dirname, 'audio'))
        for vname in tqdm(vnames):
            sname = vname[:-4] + '.npy'
            print(os.path.join(dirname, 'audio', vname))
            sound_data, _ = sf.read(os.path.join(dirname, 'audio', vname))
            if len(sound_data.shape) == 1:
                sound_data = np.expand_dims(sound_data, axis = 1)
            sound_data = torch.tensor(sound_data).transpose(0, 1).cuda().float()
            feat = self.forward(sound_data)
            np.save(os.path.join(dirname, 'afeat', self.name, sname), feat.detach().cpu())