import numpy as np
import wave
import math

def nextpow2(n):
    '''
    求最接近数据长度的2的整数次方
    An integer equal to 2 that is closest to the length of the data
    
    Eg: 
    nextpow2(2) = 1
    nextpow2(2**10+1) = 11
    nextpow2(2**20+1) = 21
    '''
    return np.ceil(np.log2(np.abs(n))).astype('long')


def berouti(SNR):
    if -5.0 <= SNR <= 20.0:
        a = 4 - SNR * 3 / 20
    else:
        if SNR < -5.0:
            a = 5
        if SNR > 20:
            a = 1
    return a


def berouti1(SNR):
    if -5.0 <= SNR <= 20.0:
        a = 3 - SNR * 2 / 20
    else:
        if SNR < -5.0:
            a = 4
        if SNR > 20:
            a = 1
    return a


def find_index(x_list):
    index_list = []
    for i in range(len(x_list)):
        if x_list[i] < 0:
            index_list.append(i)
    return index_list


def denoise_audio(input_path, output_path):
    f: wave.Wave_read = wave.open(input_path, 'rb')
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    fs = framerate

    str_data = f.readframes(nframes)
    f.close()
    x = np.frombuffer(str_data, dtype=np.short)

    len_ = 20 * fs // 1000
    PERC = 50
    len1 = len_ * PERC // 100
    len2 = len_ - len1

    Thres = 3
    Expnt = 2.0
    beta = 0.002
    G = 0.9

    # 初始化汉明窗
    win = np.hamming(len_)
    winGain = len2 / sum(win)

    # 噪声幅度计算
    nFFT = 2 * 2**(nextpow2(len_))
    noise_mean = np.zeros(nFFT)

    j = 0
    for k in range(1, 6):
        ####################
        # your code
        noise_mean += abs(np.fft.fft(win * x[j:j + len_], nFFT))
        ####################
        j = j + len_

    noise_mu = noise_mean / 5

    # 分配内存并初始化各种变量
    k = 1
    img = 1j
    x_old = np.zeros(len1)
    Nframes = len(x) // len2 - 1
    xfinal = np.zeros(Nframes * len2)

    for n in range(0, Nframes):
        # 窗口
        insign = win * x[k - 1:k + len_ - 1]
        # FFT
        spec = np.fft.fft(insign, nFFT)
        # 计算幅值
        sig = abs(spec)
        # 存储相位信息
        theta = np.angle(spec)

        # 信噪比计算
        ####################
        # your code

        SNRseg = 10 * np.log10(
            np.linalg.norm(sig, 2)**2 / np.linalg.norm(noise_mu, 2)**2)

        ####################

        if Expnt == 1.0:
            alpha = berouti1(SNRseg)
        else:
            alpha = berouti(SNRseg)

        # 谱减
        ####################
        # your code

        sub_speech = sig**Expnt - alpha * noise_mu**Expnt

        ####################

        # 当纯净信号小于噪声信号时的处理
        diffw = sub_speech - beta * noise_mu**Expnt
        z = find_index(diffw)
        if len(z) > 0:
            sub_speech[z] = beta * noise_mu[z]**Expnt

        # 更新噪声频谱
        if SNRseg < Thres:
            noise_temp = G * noise_mu**Expnt + (1 - G) * sig**Expnt
            noise_mu = noise_temp**(1 / Expnt)

        # 信号恢复
        sub_speech[nFFT // 2 + 1:nFFT] = np.flipud(sub_speech[1:nFFT // 2])

        ####################
        # your code

        x_phase = (sub_speech
                **(1 / Expnt)) * (np.array([math.cos(x) for x in theta]) + img *
                                    (np.array([math.sin(x) for x in theta])))

        ####################
        xi = np.fft.ifft(x_phase).real

        # 重叠叠加
        xfinal[k - 1:k + len2 - 1] = x_old + xi[0:len1]
        x_old = xi[0 + len1:len_]
        k = k + len2

    # 保存结果
    wf = wave.open(output_path, 'wb')
    wf.setparams(params)
    wave_data = (winGain * xfinal).astype(np.short)
    wf.writeframes(wave_data.tobytes())
    wf.close()


if __name__ == '__main__':
    denoise_audio('input.wav', 'output.wav')
