import os
from optparse import OptionParser

import numpy as np
import torch
from tqdm import tqdm

import models
import wandb
from tools.config_tools import Config


def gen_tsample(n, root_path):
    tsample = np.zeros((500, n)).astype(np.int16)
    for i in range(500):
        tsample[i] = np.random.permutation(data_num)[:n]
    np.save(f'{root_path}/tsample_{n}.npy', tsample)


def get_top(tsample, rst):
    top1 = 0.0
    top5 = 0.0
    n = tsample.shape[1]

    for i in range(500):
        idx = tsample[i]
        rsti = rst[idx][:, idx]
        assert rsti.shape[0] == n
        assert rsti.shape[1] == n
        sorti = np.sort(rsti, axis=1)
        for j in range(n):
            if rsti[j, j] == sorti[j, -1]:
                top1 += 1
            if rsti[j, j] >= sorti[j, -5]:
                top5 += 1

    top1 = top1 / 500 / n
    top5 = top5 / 500 / n
    print('Top1 accuracy for sample {} is: {}.'.format(n, top1))
    print('Top5 accuracy for sample {} is: {}.'.format(n, top5))
    wandb.log({'top1': top1})
    wandb.log({'top5': top5})


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('--config',
                      type=str,
                      help="testing configuration",
                      default="./configs/test_config.yaml")

    (opts, args) = parser.parse_args()
    assert isinstance(opts, object)
    opt = Config(opts.config)
    print(opt)

    model = models.ClassificationNet(Vinput_size = opt.v_input_size, Ainput_size = opt.a_input_size)
    ckpt = torch.load(f'checkpoints/{opt.ckpt_name}.pth',
                      map_location='cpu')
    model.load_state_dict(ckpt)
    model.cuda().eval()

    vpath = opt.vpath
    apath = opt.apath
    root_path = 'result' + '/' + opt.ckpt_name + '/' + opt.type
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    assert opt.type in opt.vpath, 'mismatch in video'
    assert opt.type in opt.apath, 'mismatch in audio'

    wandb.init(project = 'test', name = opt.ckpt_name, reinit = True, entity = "ther")

    data_num = len(os.listdir(opt.vpath))
    rst = np.zeros((data_num, data_num))
    vfeats = torch.zeros(data_num, opt.v_input_size, 10).float()
    afeats = torch.zeros(data_num, opt.a_input_size, 10).float()

    for i in tqdm(range(data_num)):
        vfeat = np.load(os.path.join(vpath, '%04d.npy' % i))
        for j in range(data_num):
            vfeats[j] = torch.from_numpy(vfeat).float().permute(1, 0)
            afeat = np.load(os.path.join(apath, '%04d.npy' % j))
            afeats[j] = torch.from_numpy(afeat).float().permute(1, 0)
        with torch.no_grad():
            out = model(vfeats.cuda(), afeats.cuda())
        rst[i] = (out[:, 1] - out[:, 0]).cpu().numpy()

    np.save(f'{root_path}/rst.npy', rst)

    print(f'Test checkpoint {opt.ckpt_name}.')

    gen_tsample(50, root_path)

    tsample = np.load(f'{root_path}/tsample_50.npy')
    get_top(tsample, rst)
