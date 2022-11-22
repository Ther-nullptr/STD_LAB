import torch
import numpy as np
from transformers import BeitForImageClassification

from .video_extractor import VideoExtractor

class BeiTExtractor(VideoExtractor):
    def __init__(self, cfg):
        super(BeiTExtractor, self).__init__(cfg)
        self.model = BeitForImageClassification.from_pretrained(cfg.local_path)
        
        self.cuda = cfg.cuda
        if cfg.cuda:
            self.model.cuda()

        self.model.eval()

    def forward(self, x: torch.Tensor):
        output = self.model.beit.forward(x)
        return output.pooler_output