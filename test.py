from optparse import OptionParser
import models
import torch
import numpy as np
import os

from tools.config_tools import Config

def gen_tsample(n):
    tsample = np.zeros((500, n)).astype(np.int16)
    for i in range(500):
        tsample[i] = np.random.permutation(804)[:n]
    np.save('tsample_{}.npy'.format(n), tsample)

def get_top(tsample, rst):
    top1 = 0.0
    top5 = 0.0
    n = tsample.shape[1]

    for i in range(500):
        idx = tsample[i]
        rsti = rst[idx][:,idx]
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

    epoch = opt.epoch
    prefix = opt.prefix
    model = models.FrameByFrame()
    ckpt = torch.load(f'checkpoints/{prefix}_state_epoch{epoch}.pth', map_location='cpu')
    model.load_state_dict(ckpt)
    model.cuda().eval()

    vpath = opt.vpath
    apath = opt.apath

    rst = np.zeros((804, 804))
    vfeats = torch.zeros(804, 512, 10).float()
    afeats = torch.zeros(804, 128, 10).float()

    for i in range(804):
        vfeat = np.load(os.path.join(vpath, '%04d.npy'%i))
        for j in range(804):
            vfeats[j] = torch.from_numpy(vfeat).float().permute(1, 0)
            afeat = np.load(os.path.join(apath, '%04d.npy'%j))
            afeats[j] = torch.from_numpy(afeat).float().permute(1, 0)
        with torch.no_grad():
            out = model(vfeats.cuda(), afeats.cuda())
        rst[i] = (out[:,1] - out[:,0]).cpu().numpy()
        print(i)

    np.save('rst_epoch{}.npy'.format(epoch), rst)

    print('Test checkpoint epoch {}.'.format(epoch))

    gen_tsample(50)

    tsample = np.load('tsample_{}.npy'.format(50))
    get_top(tsample, rst)
