import sys

sys.path.append('..')
from tools.config_tools import Config
import models

if __name__ == '__main__':
    print(dir(models))
    model_class = getattr(models, "ResNet101Extractor")
    cfg = Config('../configs/video_extractor.yaml')
    model = model_class(cfg)
    print(model)
