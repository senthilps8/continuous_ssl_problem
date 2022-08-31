#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import builtins
import os
import sys
import shutil
import time
import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import builtins
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import apex

import datasets.imagelistdataset


def main_worker(gpu, ngpus_per_node, args):
    for i in args.lincls.keys():
        main_dataset_worker(gpu, ngpus_per_node, args, dataset_i=i)


def main_dataset_worker(gpu, ngpus_per_node, args, dataset_i=0):
    best_acc1 = 0

    cudnn.benchmark = True
    args.environment.gpu = gpu

    if args.environment.gpu is not None:
        print("Use GPU: {} for training".format(args.environment.gpu))

    if args.environment.distributed and not torch.distributed.is_initialized():
        if args.environment.dist_url == "env://" and args.environment.rank == -1:
            args.environment.rank = int(os.environ["RANK"])
        if args.environment.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.environment.rank = args.environment.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.environment.dist_backend,
                                init_method=args.environment.dist_url,
                                world_size=args.environment.world_size,
                                rank=args.environment.rank)
        args.environment.workers = int(
            (args.environment.workers + ngpus_per_node - 1) / ngpus_per_node)

    os.makedirs(os.path.join(
        args.logging.submitit_dir,
        args.logging.name + '_LinCls_{:010.4f}_{}{}'.format(
            args.lincls[dataset_i].eval_params.resume_epoch,
            args.lincls[dataset_i].optim.method,
            args.lincls[dataset_i].eval_params.suffix)),
                exist_ok=True)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=os.path.join(
            args.logging.submitit_dir,
            args.logging.name + '_LinCls_{:010.4f}_{}{}'.format(
                args.lincls[dataset_i].eval_params.resume_epoch,
                args.lincls[dataset_i].optim.method,
                args.lincls[dataset_i].eval_params.suffix),
            'rank{:02d}.out'.format(args.environment.rank)),
        filemode='a')

    orig_print = builtins.print

    def new_print(*out, **kwargs):
        if not (args.environment.multiprocessing_distributed
                and args.environment.gpu != 0):
            orig_print(*out, **kwargs)
        logger.info(*out)

    builtins.print = new_print

    # create model
    print("=> creating model '{}'".format(args.model.backbone.arch))
    model = models.__dict__[args.model.backbone.arch](
        num_classes=args.lincls[dataset_i].eval_params.num_classes)
    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    # init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()
    if args.lincls[dataset_i].optim.normalize:

        class L2Norm(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return F.normalize(x, 2)

        model.fc = nn.Sequential(L2Norm(), model.fc)

    # load from pre-trained, before DistributedDataParallel constructor
    os.makedirs(os.path.join(args.logging.result_dir, args.logging.name),
                exist_ok=True)

    if args.lincls[dataset_i].eval_params.resume_epoch < 0:
        ckpt_fname = os.path.join(args.logging.ckpt_dir, args.logging.name,
                                  'checkpoint_{:04d}.pth')
        print('Checking format {}'.format(ckpt_fname))
        model_name = ''
        last_epoch = -1
        for i in range(500 - 1, -1, -1):
            if os.path.exists(ckpt_fname.format(i)):
                model_name = ckpt_fname.format(i)
                last_epoch = i
                break
    elif args.lincls[dataset_i].eval_params.resume_epoch % 1 == 0:
        ckpt_fname = os.path.join(args.logging.ckpt_dir, args.logging.name,
                                  'checkpoint_{:04d}.pth')
        i = args.lincls[dataset_i].eval_params.resume_epoch
        model_name = ckpt_fname.format(i)
        last_epoch = i
    else:
        ckpt_fname = os.path.join(args.logging.ckpt_dir, args.logging.name,
                                  'checkpoint_{:010.4f}.pth')
        i = args.lincls[dataset_i].eval_params.resume_epoch
        model_name = ckpt_fname.format(i)
        last_epoch = i

    log_string = args.logging.name + '_LinCls_epoch_{:010.4f}'.format(
        last_epoch)

    if model_name != '':
        if not os.path.isfile(model_name):
            print(f'{model_name} does not exist')
            return
        print("=> loading checkpoint '{}'".format(model_name))
        checkpoint = torch.load(model_name, map_location="cpu")

        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q'
                            ) and not k.startswith('module.encoder_q.fc'):
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]

            if k.startswith('module.encoder'
                            ) and not k.startswith('module.encoder.fc'):
                state_dict[k[len("module.encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        args.lincls[dataset_i].optim.start_epoch = 0
        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"} or set(
            msg.missing_keys) == {"fc.1.weight", "fc.1.bias"}

        print("=> loaded pre-trained model '{}'".format(model_name))
    else:
        print("=> no checkpoint found")

    result_fname = model_name.replace(
        os.path.join(args.logging.ckpt_dir, args.logging.name),
        os.path.join(args.logging.result_dir, args.logging.name))

    if args.environment.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.environment.gpu is not None:
            torch.cuda.set_device(args.environment.gpu)
            model.cuda(args.environment.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.lincls[dataset_i].optim.batch_size = int(
                args.lincls[dataset_i].optim.batch_size / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.environment.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.environment.gpu is not None:
        torch.cuda.set_device(args.environment.gpu)
        model = model.cuda(args.environment.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    cudnn.benchmark = True
    # Data loading code
    trainfname = args.lincls[dataset_i].data.train_filelist
    valfname = args.lincls[dataset_i].data.val_filelist
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.imagelistdataset.ImageListStandardDataset(
        trainfname,
        transforms.Compose([
            transforms.RandomResizedCrop(args.lincls[dataset_i].data.insize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    if args.environment.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.lincls[dataset_i].optim.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.environment.workers,
        pin_memory=True,
        sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        datasets.imagelistdataset.ImageListStandardDataset(
            valfname,
            transforms.Compose([
                transforms.Resize(
                    int(args.lincls[dataset_i].data.insize * 256 /
                        float(224))),
                transforms.CenterCrop(args.lincls[dataset_i].data.insize),
                transforms.ToTensor(),
                normalize,
            ])),
        batch_size=args.lincls[dataset_i].optim.batch_size,
        shuffle=False,
        num_workers=args.environment.workers,
        pin_memory=True)

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias
    optimizer = torch.optim.SGD(
        parameters,
        args.lincls[dataset_i].optim.lr,
        momentum=args.lincls[dataset_i].optim.momentum,
        weight_decay=args.lincls[dataset_i].optim.weight_decay)
    if args.lincls[dataset_i].optim.method == 'lars':
        optimizer = apex.parallel.LARC.LARC(optimizer,
                                            trust_coefficient=0.001,
                                            clip=False)

    # optionally resume from a checkpoint
    if args.environment.resume:
        resume_fn = model_name + args.lincls[
            dataset_i].eval_params.suffix + '.lincls'
        if os.path.isfile(resume_fn):
            print("=> loading checkpoint '{}'".format(resume_fn))
            try:
                checkpoint = torch.load(resume_fn)
            except:
                loc = 'cuda:{}'.format(args.environment.gpu)
                checkpoint = torch.load(resume_fn, map_location=loc)
            args.lincls[dataset_i].optim.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if not isinstance(best_acc1, torch.Tensor):
                best_acc1 = torch.tensor(best_acc1)
            best_acc1 = best_acc1.to(args.environment.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                resume_fn, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(resume_fn))

    for epoch in range(args.lincls[dataset_i].optim.start_epoch,
                       args.lincls[dataset_i].optim.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args, dataset_i=dataset_i)

        # train for one epoch
        train(train_loader,
              model,
              criterion,
              optimizer,
              epoch,
              args,
              writer=None)

    # evaluate on validation set
    stats = validate_with_stats(val_loader,
                                model,
                                criterion,
                                args.lincls[dataset_i].optim.epochs - 1,
                                args,
                                writer=None)
    acc1 = stats['acc1']

    if not args.environment.multiprocessing_distributed or (
            args.environment.multiprocessing_distributed
            and args.environment.rank == 0):
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        save_checkpoint(
            {
                'epoch': args.lincls[dataset_i].optim.epochs,
                'arch': args.model.backbone.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'acc1': acc1,
                'optimizer': optimizer.state_dict(),
                'stats': stats,
            },
            is_best,
            filename=model_name + args.lincls[dataset_i].eval_params.suffix +
            '.lincls')
        save_checkpoint(stats,
                        is_best,
                        filename=result_fname +
                        args.lincls[dataset_i].eval_params.suffix + '.lincls')
        if args.logging.use_wandb:
            import wandb
            wandb.init(project='streamssl',
                       name=log_string,
                       dir=args.logging.wandb_dir,
                       entity=args.logging.wandb_username,
                       config=dict(args))
            wandb.log({'lincls_acc': stats['acc1']})


def train(train_loader, model, criterion, optimizer, epoch, args, writer=None):
    batch_time = AverageMeter('Time', ':6.3f', tbname='train/time')
    data_time = AverageMeter('Data', ':6.3f', tbname='train/datatime')
    losses = AverageMeter('Loss', ':.4e', tbname='train/loss')
    top1 = AverageMeter('Acc@1', ':6.2f', tbname='train/top1')
    top5 = AverageMeter('Acc@5', ':6.2f', tbname='train/top5')
    progress = ProgressMeter(len(train_loader),
                             [batch_time, data_time, losses, top1, top5],
                             prefix="Epoch: [{}]".format(epoch),
                             tbwriter=writer)
    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()

    end = time.time()
    for i, out in enumerate(train_loader):
        (images, target) = out['input'], out['target']
        # measure data loading time
        torch.distributed.barrier()
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.environment.print_freq == 0:
            progress.display(i)
            progress.tbwrite(epoch * len(train_loader.dataset) //
                             args.optim.batch_size + i)
            sys.stdout.flush()


def validate_with_stats(val_loader,
                        model,
                        criterion,
                        epoch,
                        args,
                        writer=None):
    batch_time = AverageMeter('Time', ':6.3f', tbname='val/time')
    losses = AverageMeter('Loss', ':.4e', tbname='val/loss')
    top1 = AverageMeter('Acc@1', ':6.2f', tbname='val/top1')
    top5 = AverageMeter('Acc@5', ':6.2f', tbname='val/top5')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5],
                             prefix='Test: ',
                             tbwriter=writer)

    # switch to evaluate mode
    model.eval()
    all_stats = {}
    targets = []
    preds = []

    with torch.no_grad():
        end = time.time()
        for i, out in enumerate(val_loader):
            (images, target) = out['input'], out['target']
            targets.append(target.numpy())
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            _, pred = output.topk(5, 1, True, True)
            preds.append(pred.cpu().numpy())

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.environment.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        progress.sync_distributed()
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1,
                                                                    top5=top5))
        progress.tbwrite(0)
        all_stats['acc1'] = top1.avg
        all_stats['acc5'] = top5.avg
        all_stats['preds'] = preds
        all_stats['targets'] = targets
    return all_stats


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename[:-6] + 'bestlincls')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', tbname=None):
        self.name = name
        self.fmt = fmt
        self.tbname = tbname
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
            if meter.tbname is None:
                meter.tbname = meter.name
            tag = meter.tbname
            sclrval = val
            out[tag] = sclrval
        return out

    def sync_distributed(self):
        if torch.distributed.is_initialized():
            for meter in self.meters:
                sum_count = (meter.sum, meter.count)
                sum_count_list = [
                    None for _ in range(torch.distributed.get_world_size())
                ]
                torch.distributed.all_gather_object(sum_count_list, sum_count)
                meter.sum = sum([s for s, c in sum_count_list])
                meter.count = sum([c for s, c in sum_count_list])
                meter.avg = meter.sum / meter.count


def adjust_learning_rate(optimizer, epoch, args, dataset_i=0):
    """Decay the learning rate based on schedule"""
    lr = args.lincls[dataset_i].optim.lr
    if args.lincls[dataset_i].optim.cos:
        lr *= 0.5 * (1. + math.cos(
            math.pi * epoch / args.lincls[dataset_i].optim.epochs))
    else:
        for milestone in args.lincls[dataset_i].optim.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
