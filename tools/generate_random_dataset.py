import os
import random
import numpy as np

def padding_zero(num, length = 4):
    num_str = str(num)
    for _ in range(length - len(num_str)):
        num_str = '0' + num_str
    return num_str


def soft_connect(src_dir, tgt_dir, extractor_name, index_list, padding = 4):
    src_dir_w_model = f'{src_dir}/{extractor_name}'
    tgt_dir_w_model = f'{tgt_dir}/{extractor_name}'

    if not os.path.exists(tgt_dir_w_model):
        os.system(f'mkdir -p {tgt_dir_w_model}')

    for i, item in enumerate(index_list):
        os.system(f'ln -s {src_dir_w_model}/{padding_zero(item, padding)}.npy {tgt_dir_w_model}/{padding_zero(i, padding)}.npy')

if __name__ == '__main__':
    afeat_original_dir = '/root/kyzhang/yjwang/InclusiveFL2/data/Dataset/Train/afeat'
    vfeat_original_dir = '/root/kyzhang/yjwang/InclusiveFL2/data/Dataset/Train/vfeat'

    audio_original_dir = '/root/kyzhang/yjwang/InclusiveFL2/data/Dataset/Train/audio'
    video_original_dir = '/root/kyzhang/yjwang/InclusiveFL2/data/Dataset/Train/video'

    label = list(np.arange(3339))
    label_dev = label[0::10] # 10%
    label_train = list(set(label) - set(label_dev))

    afeat_train_dir = '/root/kyzhang/yjwang/InclusiveFL2/data/Dataset/Train_Part/afeat'
    vfeat_train_dir = '/root/kyzhang/yjwang/InclusiveFL2/data/Dataset/Train_Part/vfeat'

    audio_train_dir = '/root/kyzhang/yjwang/InclusiveFL2/data/Dataset/Train_Part/audio'
    video_train_dir = '/root/kyzhang/yjwang/InclusiveFL2/data/Dataset/Train_Part/video'

    afeat_dev_dir = '/root/kyzhang/yjwang/InclusiveFL2/data/Dataset/Dev/afeat'
    vfeat_dev_dir = '/root/kyzhang/yjwang/InclusiveFL2/data/Dataset/Dev/vfeat'

    audio_dev_dir = '/root/kyzhang/yjwang/InclusiveFL2/data/Dataset/Dev/audio'
    video_dev_dir = '/root/kyzhang/yjwang/InclusiveFL2/data/Dataset/Dev/video'

    soft_connect(afeat_original_dir, afeat_train_dir, 'vggish-quant', label_train, padding = 4)
    soft_connect(afeat_original_dir, afeat_dev_dir, 'vggish-quant', label_dev, padding = 4)
    soft_connect(vfeat_original_dir, vfeat_train_dir, 'BeiTExtractor', label_train, padding = 4)
    soft_connect(vfeat_original_dir, vfeat_dev_dir, 'BeiTExtractor', label_dev, padding = 4)

