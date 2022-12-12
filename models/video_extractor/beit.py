import torch
import numpy as np
from transformers import BeitForImageClassification
from numpy.random import normal

from .video_extractor import VideoExtractor

class BeiTExtractor(VideoExtractor):
    def __init__(self, cfg):
        super(BeiTExtractor, self).__init__(cfg)
        self.model = BeitForImageClassification.from_pretrained(cfg.local_path)
        
        self.cuda = cfg.cuda
        if cfg.cuda:
            self.model.cuda()

        self.model.eval()
        self.use_noise = cfg.use_noise
        self.mean = cfg.mean / 255
        self.std = cfg.std / 255

    def forward(self, x: torch.Tensor):
        if self.use_noise:
            x = x + torch.Tensor(normal(loc = self.mean , scale = self.std, size = x.shape)).cuda()
        output = self.model.beit.forward(x)
        return output.pooler_output