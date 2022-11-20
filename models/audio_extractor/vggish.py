import torch

from .vggish_model.vggish import VGGish
from .audio_extractor import AudioExtractor

class VggishExtractor(AudioExtractor):
    def __init__(self, cfg):
        super(VggishExtractor, self).__init__(cfg)
        if cfg.pretrained:
            self.model = VGGish()
            state_dict = torch.load(cfg.param_path)
            self.model.load_state_dict(state_dict)
        else:
            self.model = VGGish()

        self.cuda = cfg.cuda
        if cfg.cuda:
            self.model.cuda()

        self.model.eval()

    def forward(self, x: torch.Tensor):
        return self.model.forward(x)
