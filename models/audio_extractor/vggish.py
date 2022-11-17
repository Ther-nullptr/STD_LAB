from torchvggish.hubconf import vggish
from .audio_extractor import AudioExtractor

class VggishExtractor(AudioExtractor):
    def __init__(self, cfg):
        super(VggishExtractor, self).__init__(cfg)
        self.model = vggish(pretrained=cfg.pretrained)
        self.model.eval()

    def forward(self, x):
        return self.model.forward(x)
