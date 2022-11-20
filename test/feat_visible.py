import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    audio_feat = np.load('/root/kyzhang/yjwang/InclusiveFL2/test/audio_feat.npy')
    video_feat = np.load('/root/kyzhang/yjwang/InclusiveFL2/test/video_feat.npy')
    print(audio_feat.shape)
    print(video_feat.shape)

    plt.imshow(audio_feat)
    plt.plot()
    plt.savefig('audio_feat.png')

    plt.imshow(video_feat)
    plt.plot()
    plt.savefig('video_feat.png')
    