import soundfile as sf
import numpy as np

if __name__ == '__main__':
    data1, samplerate = sf.read('/root/kyzhang/yjwang/InclusiveFL2/test/noise/input.wav')
    data2, samplerate = sf.read('/root/kyzhang/yjwang/InclusiveFL2/test/noise/output.wav')
    min_length = min(len(data1), len(data2))

    data1 = data1[:min_length,:]
    data2 = data2[:min_length,:]

    # plot the distribution of noise
    import matplotlib.pyplot as plt
    plt.hist((data1.astype('float32') - data2.astype('float32')).flatten(), bins=1000, range=[-0.5,0.5])
    plt.savefig('noise_distribution_audio.png')

    from scipy.stats import norm
    mean, std = norm.fit((data1.astype('float32') - data2.astype('float32')).flatten())
    print(mean, std**2)


