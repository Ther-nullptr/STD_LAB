import numpy as np
import sys

sys.path.append('..')
from models.video_extractor import *
from tools.config_tools import Config

if __name__ == '__main__':
    path = '/mnt/c/Users/86181/Desktop/STD-project/test/input_mp4_file.mp4'
    cfg = Config('../configs/video_extractor.yaml')
    model = ResNet34Extractor(cfg)
    output = model.extract_video(path)
