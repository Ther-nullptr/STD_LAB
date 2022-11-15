import numpy as np
import sys

sys.path.append('..')
from torchvggish.model.vggish import *

if __name__ == '__main__':
    model = VGGish()
    print(model)
    model.eval()

    raw_wav = 'input_wav_file.wav'
    output_1 = model(raw_wav)
    print(output_1.shape)
    
    feat_wav = np.load('input_wav_file.npy')
    output_2 = model(feat_wav, 16000)
    print(output_2.shape)
