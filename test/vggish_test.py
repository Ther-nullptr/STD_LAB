import numpy as np
import sys

sys.path.append('..')
from models.audio_extractor import *
from tools.config_tools import Config

if __name__ == '__main__':
    raw_wav = 'input_wav_file.wav'
    cfg = Config('../configs/audio_extractor.yaml')
    model = VggishExtractor(cfg)
    output_1 = model(raw_wav)
    print(output_1.shape)
