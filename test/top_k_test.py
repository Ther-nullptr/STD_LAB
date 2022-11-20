import numpy as np

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

if __name__ == '__main__':
    tsample = np.load('/root/kyzhang/yjwang/InclusiveFL2/result/modelFrameByFramebatchSize64lr0.001vpathresnet-101apathvggish-quantmax_epochs100/Clean/tsample_50.npy')
    rst = np.load('/root/kyzhang/yjwang/InclusiveFL2/result/modelFrameByFramebatchSize64lr0.001vpathresnet-101apathvggish-quantmax_epochs100/Clean/rst.npy')
    get_top(tsample, rst)
