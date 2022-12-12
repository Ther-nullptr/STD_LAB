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
    root_path = '/root/kyzhang/yjwang/InclusiveFL2/data/Dataset'

    afeat_original_dir = f'{root_path}/Train/afeat'
    vfeat_original_dir = f'{root_path}/Train/vfeat'

    audio_original_dir = f'{root_path}/Train/audio'
    video_original_dir = f'{root_path}/Train/video'

    label = list(np.arange(3339))
    label_dev = label[0::10] # 10%
    label_train = list(set(label) - set(label_dev))

    afeat_train_dir = f'{root_path}/Train_Part/afeat'
    vfeat_train_dir = f'{root_path}/Train_Part/vfeat'

    audio_train_dir = f'{root_path}/Train_Part/audio'
    video_train_dir = f'{root_path}/Train_Part/video'

    afeat_dev_dir = f'{root_path}/Dev/afeat'
    vfeat_dev_dir = f'{root_path}/Dev/vfeat'

    audio_dev_dir = f'{root_path}/Dev/audio'
    video_dev_dir = f'{root_path}/Dev/video'

    soft_connect(afeat_original_dir, afeat_train_dir, 'VggishExtractor_noise', label_train, padding = 4)
    soft_connect(afeat_original_dir, afeat_dev_dir, 'VggishExtractor_noise', label_dev, padding = 4)
    soft_connect(vfeat_original_dir, vfeat_train_dir, 'BeiTExtractor_noise', label_train, padding = 4)
    soft_connect(vfeat_original_dir, vfeat_dev_dir, 'BeiTExtractor_noise', label_dev, padding = 4)

