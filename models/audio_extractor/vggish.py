import torch
from numpy.random import normal

from .vggish_model.vggish import VGGish
from .audio_extractor import AudioExtractor

class VggishExtractor(AudioExtractor):
    def __init__(self, cfg):
        super(VggishExtractor, self).__init__(cfg)
        self.model = VGGish(pretrained=True)

        self.cuda = cfg.cuda
        if cfg.cuda:
            self.model.cuda()

        self.model.eval()
        self.use_noise = cfg.use_noise
        self.mean = cfg.mean
        self.std = cfg.std

    def forward(self, x, fs):
        if self.use_noise:
            x = x + torch.Tensor(normal(loc = self.mean , scale = self.std, size = x.shape)).cuda()
        return self.model.forward(x, fs)
