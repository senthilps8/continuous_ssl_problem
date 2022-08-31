import os
import sys
import math
import glob
import re
import time

import torch
import torch.distributed
import numpy as np
from simsiam import concat_all_gather


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    distributed = torch.distributed.is_available(
    ) and torch.distributed.is_initialized()
    if not distributed or (distributed and torch.distributed.get_rank() == 0):
        torch.save(state, filename)
        print("=> saved checkpoint '{}' (epoch {})".format(
            filename, state['epoch']))


def resume_from_checkpoint(ckpt_fname, modules, args):
    print("=> loading checkpoint '{}'".format(ckpt_fname))
    if args.environment.gpu == '' or args.environment.gpu is None:
        checkpoint = torch.load(ckpt_fname, map_location='cpu')
    else:
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(args.environment.gpu)
        checkpoint = torch.load(ckpt_fname, map_location=loc)

    # Load state dict
    for k in modules:
        modules[k].load_state_dict(checkpoint[k])
    args.optim.start_epoch = checkpoint['epoch']
    if 'batch_i' in checkpoint:
        args.optim.start_batch_idx = checkpoint['batch_i'] + 1
        print("=> loaded checkpoint '{}' (epoch {}, batch {})".format(
            ckpt_fname, checkpoint['epoch'], checkpoint['batch_i']))
    else:
        print("=> loaded checkpoint '{}' (epoch {})".format(
            ckpt_fname, checkpoint['epoch']))
    return args


def resume(modules, args):
    all_ckpt_fnames = glob.glob(
        os.path.join(args.logging.ckpt_dir, args.logging.name,
                     'checkpoint_*.pth'))
    if not all_ckpt_fnames:
        return

    # Find last checkpoint
    epochs = [
        float(
            re.match('checkpoint_(\d+\.*\d*).pth',
                     fn.split('/')[-1]).group(1)) for fn in all_ckpt_fnames
    ]
    ckpt_fname = all_ckpt_fnames[np.argsort(-np.array(epochs))[-1]]

    # Load checkpoint
    resume_from_checkpoint(ckpt_fname, modules, args)


def adjust_learning_rate(optimizer, epoch, args, epoch_size=None):
    """Decay the learning rate based on schedule"""
    init_lr = args.optim.lr
    if args.optim.lr_schedule.type == 'constant':
        cur_lr = init_lr

    elif args.optim.lr_schedule.type == 'cos':
        cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.optim.epochs))

    elif args.optim.lr_schedule.type == 'triangle':
        T = args.optim.lr_schedule.period
        t = (epoch * epoch_size) % T
        if t < T / 2:
            cur_lr = args.optim.lr + t / (T / 2.) * (args.optim.lr_schedule.max_lr - args.optim.lr)
        else:
            cur_lr = args.optim.lr + (T-t) / (T / 2.) * (args.optim.lr_schedule.max_lr - args.optim.lr)

    else:
        raise ValueError('LR schedule unknown.')

    if args.optim.lr_schedule.exit_decay > 0:
        start_decay_epoch = args.optim.epochs * (1. - args.optim.lr_schedule.exit_decay)
        if epoch > start_decay_epoch:
            mult = 0.5 * (1. + math.cos(math.pi * (epoch - start_decay_epoch) / (args.optim.epochs - start_decay_epoch)))
            cur_lr = cur_lr * mult

    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr
    return cur_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', tbname=''):
        self.name = name
        self.tbname = tbname
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class WindowAverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, k=250, fmt=':f', tbname=''):
        self.name = name
        self.tbname = tbname
        self.fmt = fmt
        self.k = k
        self.reset()

    def reset(self):
        from collections import deque
        self.vals = deque(maxlen=self.k)
        self.counts = deque(maxlen=self.k)
        self.val = 0
        self.avg = 0

    def update(self, val, n=1):
        self.vals.append(val)
        self.counts.append(n)
        self.val = val
        self.avg = sum([v * c for v, c in zip(self.vals, self.counts)]) / sum(
            self.counts)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", tbwriter=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.tbwriter = tbwriter

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        sys.stdout.flush()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

    def tbwrite(self, batch):
        if self.tbwriter is None:
            return
        scalar_dict = self.tb_scalar_dict()
        for k, v in scalar_dict.items():
            self.tbwriter.add_scalar(k, v, batch)

    def tb_scalar_dict(self):
        out = {}
        for meter in self.meters:
            val = meter.avg
            if not meter.tbname:
                meter.tbname = meter.name
                tag = meter.tbname
                sclrval = val
                out[tag] = sclrval
        return out



class CheckpointManager:
    def __init__(self,
                 modules,
                 ckpt_dir,
                 epoch_size,
                 epochs,
                 save_freq=None,
                 save_freq_mints=None):
        self.modules = modules
        self.ckpt_dir = ckpt_dir
        self.epoch_size = epoch_size
        self.epochs = epochs
        self.save_freq = save_freq
        self.save_freq_mints = save_freq_mints
        self.retain_num_ckpt = 0

        self.time = time.time()
        self.distributed = torch.distributed.is_available(
        ) and torch.distributed.is_initialized()
        self.world_size = torch.distributed.get_world_size(
        ) if self.distributed else 1
        self.rank = torch.distributed.get_rank() if self.distributed else 0

        os.makedirs(os.path.join(self.ckpt_dir), exist_ok=True)

    def resume(self):
        ckpt_fname = os.path.join(self.ckpt_dir, 'checkpoint_latest.pth')
        start_epoch = 0
        if os.path.isfile(ckpt_fname):
            checkpoint = torch.load(ckpt_fname, map_location='cpu')

            # Load state dict
            for k in self.modules:
                self.modules[k].load_state_dict(checkpoint[k])
            start_epoch = checkpoint['epoch']
            print("=> loaded checkpoint '{}' (epoch {})".format(
                ckpt_fname, checkpoint['epoch']))
        return start_epoch

    def timed_checkpoint(self, save_dict=None):
        t = time.time() - self.time
        t_all = [t for _ in range(self.world_size)]
        if self.world_size > 1:
            torch.distributed.all_gather_object(t_all, t)
        if min(t_all) > self.save_freq_mints * 60:
            self.time = time.time()
            ckpt_fname = os.path.join(self.ckpt_dir, 'checkpoint_latest.pth')

            state = self.create_state_dict(save_dict)
            if self.rank == 0:
                save_checkpoint(state, is_best=False, filename=ckpt_fname)

    def midway_epoch_checkpoint(self, epoch, batch_i, save_dict=None):
        if ((batch_i + 1) / float(self.epoch_size) % self.save_freq) < (
                batch_i / float(self.epoch_size) % self.save_freq):
            ckpt_fname = os.path.join(self.ckpt_dir,
                                      'checkpoint_{:010.4f}.pth')
            ckpt_fname = ckpt_fname.format(epoch +
                                           batch_i / float(self.epoch_size))

            state = self.create_state_dict(save_dict)
            if self.rank == 0:
                save_checkpoint(state, is_best=False, filename=ckpt_fname)
                ckpt_fname = os.path.join(self.ckpt_dir,
                                          'checkpoint_latest.pth')
                save_checkpoint(state, is_best=False, filename=ckpt_fname)

    def end_epoch_checkpoint(self, epoch, save_dict=None):
        if (epoch % self.save_freq
                == 0) or self.save_freq < 1 or epoch == self.epochs:
            ckpt_fname = os.path.join(self.ckpt_dir, 'checkpoint_{:04d}.pth')
            ckpt_fname = ckpt_fname.format(epoch)

            state = self.create_state_dict(save_dict)
            if self.rank == 0:
                save_checkpoint(state, is_best=False, filename=ckpt_fname)
                ckpt_fname = os.path.join(self.ckpt_dir,
                                          'checkpoint_latest.pth')
                save_checkpoint(state, is_best=False, filename=ckpt_fname)

            if self.retain_num_ckpt > 0:
                ckpt_fname = os.path.join(self.ckpt_dir,
                                          'checkpoint_{:04d}.pth')
                ckpt_fname = ckpt_fname.format(epoch - self.save_freq *
                                               (self.retain_num_ckpt + 1))
                if os.path.exists(ckpt_fname):
                    os.remove(ckpt_fname.format(ckpt_fname))

    def create_state_dict(self, save_dict):
        state = {k: self.modules[k].state_dict() for k in self.modules}
        if save_dict is not None:
            state.update(save_dict)
        return state

    def checkpoint(self, epoch, batch_i=None, save_dict=None):
        if batch_i is None:
            self.end_epoch_checkpoint(epoch, save_dict)
        else:
            if batch_i % 100 == 0:
                self.timed_checkpoint(save_dict)
            self.midway_epoch_checkpoint(epoch, batch_i, save_dict=save_dict)


def shuffle_batch(x1, x2):
    """
    Batch shuffle, for making use of BatchNorm.
    *** Only support DistributedDataParallel (DDP) model. ***
    """
    if not torch.distributed.is_initialized():
        batch_size_all = x1.shape[0]
        idx_shuffle = torch.randperm(batch_size_all).to(x1.device)
        return x1[idx_shuffle], x2[idx_shuffle]

    else:
        # gather from all gpus
        batch_size_this = x1.shape[0]
        x1_gather = concat_all_gather(x1)
        x2_gather = concat_all_gather(x2)
        batch_size_all = x1_gather.shape[0]

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).to(x1.device)

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        num_gpus = batch_size_all // batch_size_this
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        # shuffle
        return x1_gather[idx_this], x2_gather[idx_this]
