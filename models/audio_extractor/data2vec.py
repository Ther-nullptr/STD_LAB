import torch
import numpy as np
from transformers import Wav2Vec2Processor, Data2VecAudioForCTC

from .audio_extractor import AudioExtractor

class Data2vecExtractor(AudioExtractor):
    def __init__(self, cfg):
        super(Data2vecExtractor, self).__init__(cfg)
        self.model = Data2VecAudioForCTC.from_pretrained(cfg.local_path)
        
        self.cuda = cfg.cuda
        if cfg.cuda:
            self.model.cuda()

        self.model.eval()

    def forward(self, x: torch.Tensor):
        output = self.model.data2vec_audio.forward(x)
        return output.pooler_output