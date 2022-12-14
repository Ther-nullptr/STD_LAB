#!/usr/bin/env python

from __future__ import print_function

import logging
import os
import random
import sys
import time
from datetime import datetime
from optparse import OptionParser

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR as LR_Policy

import models.match_net
import wandb
from dataset import VideoFeatDataset as dset
from tools import utils
from tools.config_tools import Config

logger = logging.getLogger(__name__)

# (batchsize, features, sequence)
def get_triplet(vfeat, afeat):
    vfeat_var = vfeat
    afeat_p_var = afeat
    orders = np.arange(vfeat.size(0)).astype('int32')
    negetive_orders = orders.copy()

    for i in range(len(negetive_orders)):
        index_list = list(range(i))
        index_list.extend(list(range(len(negetive_orders))[i + 1:]))
        negetive_orders[i] = index_list[random.randint(0, len(negetive_orders) - 2)]

    afeat_n_var = afeat[torch.from_numpy(negetive_orders).long()].clone()
    return vfeat_var, afeat_p_var, afeat_n_var


# training function for metric learning
def train(train_loader, model, criterion, optimizer, epoch, opt):
    """
    train for one epoch on the training set
    """
    batch_time = utils.AverageMeter()
    audio_loss = utils.AverageMeter()
    video_loss = utils.AverageMeter()

    model.train()
    end = time.time()

    for i, (vfeat, afeat, label) in enumerate(train_loader):
        bz = vfeat.size()[0]
        orders = np.arange(bz).astype('int32')
        shuffle_orders = orders.copy()
        np.random.shuffle(shuffle_orders)

        # transpose the feats
        vfeat = vfeat.transpose(2, 1)  #! [batch * 2, 512, 10]
        afeat = afeat.transpose(2, 1)  #! [batch * 2, 128, 10]

        # put the data into Variable
        vfeat_var = Variable(vfeat)
        afeat_var = Variable(afeat)
        label_var = Variable(label)

        # if you have gpu, then shift data to GPU
        if opt.cuda:
            vfeat_var = vfeat_var.cuda()
            afeat_var = afeat_var.cuda()
            label_var = label_var.cuda()

        aprob, vprob = model(vfeat_var, afeat_var)
        label = label.type(torch.LongTensor).cuda()
        aloss = criterion(aprob, label)
        vloss = criterion(vprob, label)

        audio_loss.update(aloss.item(), afeat.size(0))
        video_loss.update(vloss.item(), vfeat.size(0))

        a_predict_var = torch.argmax(aprob, dim = 1)
        v_predict_var = torch.argmax(vprob, dim = 1)
        a_train_acc = sum(a_predict_var == label_var) / len(label_var)
        v_train_acc = sum(v_predict_var == label_var) / len(label_var)

        ##############################
        # compute gradient and do sgd
        ##############################
        optimizer.zero_grad()
        loss = aloss + vloss
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % opt.print_freq == 0:
            log_str = 'Epoch: [{0}][{1}/{2}]\t Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t Loss {a_loss.val:.4f} ({a_loss.avg:.4f}) {v_loss.val:.4f} ({v_loss.avg:.4f})'.format(
                epoch,
                i,
                len(train_loader),
                batch_time=batch_time,
                a_loss=audio_loss,
                v_loss=video_loss
            )
            logger.info(log_str)
            logger.info(f'audio_train_acc:{a_train_acc}, video_train_acc:{v_train_acc}')
            wandb.log({'epoch':epoch, 'loss':loss, 'a_train_acc':a_train_acc, 'v_train_acc':v_train_acc})


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('--config',
                      type=str,
                      help="training configuration",
                      default="./configs/train_config.yaml")

    (opts, args) = parser.parse_args()
    assert isinstance(opts, object)
    opt = Config(opts.config)
    print(opt)

    if opt.checkpoint_folder is None:
        opt.checkpoint_folder = 'checkpoints'

    # make dir
    if not os.path.exists(opt.checkpoint_folder):
        os.system('mkdir {0}'.format(opt.checkpoint_folder))

    # classify
    if opt.add_classify is False:
        opt.cpath = None
    train_dataset = dset(opt.vpath, opt.apath, opt.cpath, opt.supervised)

    print('number of train samples is: {0}'.format(len(train_dataset)))
    print('finished loading data')

    opt.manualSeed = random.randint(1, 10000)

    if torch.cuda.is_available() and not opt.cuda:
        print(
            "WARNING: You have a CUDA device, so you should probably run with \"cuda: True\""
        )
        torch.manual_seed(opt.manualSeed)

    else:
        if int(opt.ngpu) == 1:
            print('so we use 1 gpu to training')
            print('setting gpu on gpuid {0}'.format(opt.gpu_id))
            if opt.cuda:
                torch.cuda.set_device(int(opt.gpu_id))
                torch.cuda.manual_seed(opt.manualSeed)
                cudnn.benchmark = True

    print('Random Seed: {0}'.format(opt.manualSeed))

    # train data loader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opt.batchSize,
                                               shuffle=True,
                                               num_workers=int(opt.workers))

    # create model
    if hasattr(models.match_net, opt.model):
        model_class = getattr(models.match_net, opt.model)
        model = model_class(Vinput_size = opt.v_input_size, Ainput_size = opt.a_input_size, dropout_ratio = opt.dropout_ratio)
    else:
        raise ModuleNotFoundError(f"No implementation of {opt.model}")

    if opt.use_ckpt:
        state_dict = torch.load(opt.ckpt_path)
        model.load_state_dict(state_dict)

    criterion = torch.nn.CrossEntropyLoss()
    if opt.cuda:
        print('shift model and criterion to GPU .. ')
        model = model.cuda()
        criterion = criterion.cuda()

    # optimizer
    optimizer = optim.Adam(model.parameters(), opt.lr)

    # adjust learning rate every lr_decay_epoch
    lambda_lr = lambda epoch: opt.lr_decay**((epoch + 1) // opt.lr_decay_epoch)
    scheduler = LR_Policy(optimizer, lambda_lr)

    # logger settings
    afeat_name = os.path.split(opt.apath)[1]
    vfeat_name = os.path.split(opt.vpath)[1]
    wandb_string = 'model' + str(opt.model) + 'batchSize' + str(opt.batchSize) + 'lr' + str(opt.lr) + 'vpath' + str(vfeat_name) + 'apath' + str(afeat_name) + 'max_epochs' + str(opt.max_epochs) + 'dropout_ratio' + str(opt.dropout_ratio) + 'comment' + str(opt.comment)
    wandb.init(project = 'train', name = wandb_string, reinit = True, entity = "ther")

    logger.setLevel(level = logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(filename)s : %(lineno)s line - %(message)s")

    log_file_name = os.path.join(opt.log_folder, wandb_string)
    file_handler = logging.FileHandler(filename=f'{log_file_name}.log')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level = logging.INFO)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level = logging.INFO)
    logger.addHandler(stream_handler)

    logger.info(str(opt))

    # train for every epoch
    for epoch in range(opt.max_epochs):
        train(train_loader, model, criterion, optimizer, epoch, opt)
        scheduler.step()
        if ((epoch + 1) % opt.epoch_save) == 0:
            if not os.path.exists(f'{opt.checkpoint_folder}/{wandb_string}'):
                os.system(f'mkdir -p {opt.checkpoint_folder}/{wandb_string}')
            path_checkpoint = f'{opt.checkpoint_folder}/{wandb_string}/state_epoch{epoch + 1}.pth'
            utils.save_checkpoint(model.state_dict(), path_checkpoint)
