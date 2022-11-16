import numpy as np
import torch
import sys

sys.path.append('..')
from models import *

if __name__ == '__main__':
    framebyframe_impl = FrameByFrame()
    print(framebyframe_impl)

    wav_feat_file = '/mnt/c/Users/86181/Desktop/STD-project/test/audio_feat.npy'
    wav_feat = torch.tensor(np.load(wav_feat_file)).unsqueeze(0).transpose(1, 2)
    print(wav_feat.shape) #! [1, 128, 10]

    video_feat_file = '/mnt/c/Users/86181/Desktop/STD-project/test/video_feat.npy'
    video_feat = torch.tensor(np.load(video_feat_file)).unsqueeze(0).transpose(1, 2)
    print(video_feat.shape) #! [1, 512, 10]

    output = framebyframe_impl.forward(video_feat, wav_feat)
    print(output.shape) #! [batch, binary prob]