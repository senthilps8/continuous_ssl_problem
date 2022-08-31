#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from torch.utils.tensorboard import SummaryWriter
import submitit
import hydra.utils as hydra_utils
import hydra
from pathlib import Path
import random
import logging
import os
import copy
import warnings

warnings.filterwarnings("ignore")
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

MAIN_PID = os.getpid()
SIGNAL_RECEIVED = False

log = logging.getLogger(__name__)


def update_pythonpath_relative_hydra():
    """Update PYTHONPATH to only have absolute paths."""
    # NOTE: We do not change sys.path: we want to update paths for future instantiations
    # of python using the current environment (namely, when submitit loads the job
    # pickle).
    try:
        original_cwd = Path(hydra_utils.get_original_cwd()).resolve()
    except (AttributeError, ValueError):
        # Assume hydra is not initialized, we don't need to do anything.
        # In hydra 0.11, this returns AttributeError; later it will return ValueError
        # https://github.com/facebookresearch/hydra/issues/496
        # I don't know how else to reliably check whether Hydra is initialized.
        return
    paths = []
    for orig_path in os.environ["PYTHONPATH"].split(":"):
        path = Path(orig_path)
        if not path.is_absolute():
            path = original_cwd / path
        paths.append(path.resolve())
    os.environ["PYTHONPATH"] = ":".join([str(x) for x in paths])
    log.info('PYTHONPATH: {}'.format(os.environ["PYTHONPATH"]))


class Worker:
    def __call__(self, origargs):
        """TODO: Docstring for __call__.

        :args: TODO
        :returns: TODO

        """
        import importlib
        main_worker = importlib.import_module(
            origargs.model.main_worker).main_worker
        from main_lincls_worker import main_worker as main_lincls_worker
        import numpy as np
        import torch.multiprocessing as mp
        import torch.utils.data.distributed
        import torch.backends.cudnn as cudnn
        mp.set_start_method('spawn')

        cudnn.benchmark = True
        args = copy.deepcopy(origargs)
        np.set_printoptions(precision=3)
        if args.environment.seed == 0:
            args.environment.seed = None
        socket_name = os.popen("ip r | grep default | awk '{print $5}'").read().strip('\n')
        print("Setting GLOO and NCCL sockets IFNAME to: {}".format(socket_name))
        os.environ["GLOO_SOCKET_IFNAME"] = socket_name
        # not sure if the next line is really affect anything
        # os.environ["NCCL_SOCKET_IFNAME"] = socket_name
        args.environment.port = np.random.randint(10000, 20000)
        assert args.environment.world_size <= 1, "Only single node training implemented."

        if args.environment.slurm:
            job_env = submitit.JobEnvironment()
            args.environment.rank = job_env.global_rank
            args.environment.dist_url = f'tcp://{job_env.hostnames[0]}:{args.environment.port}'
        else:
            args.environment.dist_url = f'tcp://{args.environment.node}:{args.environment.port}'
        print('Using url {}'.format(args.environment.dist_url))

        if args.logging.log_tb:
            os.makedirs(os.path.join(args.logging.tb_dir, args.logging.name),
                        exist_ok=True)
            writer = SummaryWriter(
                os.path.join(args.logging.tb_dir, args.logging.name))
            writer.add_text('exp_dir', os.getcwd())

        if args.environment.seed is not None:
            random.seed(args.environment.seed)
            torch.manual_seed(args.environment.seed)
            warnings.warn('You have chosen to seed training. '
                          'This will turn on the CUDNN deterministic setting, '
                          'which can slow down your training considerably! '
                          'You may see unexpected behavior when restarting '
                          'from checkpoints.')
        if args.environment.gpu != '':
            warnings.warn(
                'You have chosen a specific GPU. This will completely '
                'disable data parallelism.')

        if args.environment.dist_url == "env://" and args.environment.world_size == -1:
            args.environment.world_size = int(os.environ["WORLD_SIZE"])

        args.environment.distributed = args.environment.world_size > 1 or args.environment.multiprocessing_distributed
        ngpus_per_node = torch.cuda.device_count()
        lincls_key = list(args.lincls.keys())[0]
        if args.environment.multiprocessing_distributed:
            # Since we have ngpus_per_node processes per node, the total world_size
            # needs to be adjusted accordingly
            args.environment.world_size = ngpus_per_node * args.environment.world_size
            # Use torch.multiprocessing.spawn to launch distributed processes: the
            # main_worker process function
            lincls_args = copy.deepcopy(args)
            mp.spawn(main_worker,
                     nprocs=ngpus_per_node,
                     args=(ngpus_per_node, args))
            if args.lincls[lincls_key].eval_params.resume_epoch >= 0:
                mp.spawn(main_lincls_worker,
                         nprocs=ngpus_per_node,
                         args=(ngpus_per_node, args))
        else:
            # Simply call main_worker function
            lincls_args = copy.deepcopy(args)
            main_worker(args.environment.gpu, ngpus_per_node, args)
            if args.lincls[lincls_key].eval_params.resume_epoch >= 0:
                main_lincls_worker(args.environment.gpu, ngpus_per_node, args)

    def checkpoint(self, *args,
                   **kwargs) -> submitit.helpers.DelayedSubmission:
        return submitit.helpers.DelayedSubmission(
            Worker(), *args, **kwargs)  # submits to requeuing


def jobs_running():
    return [jobname for jobname in
            os.popen('squeue -o %j').read().split("\n")]


@hydra.main(config_path='./configs/simsiam/', config_name='config')
def main(args):
    update_pythonpath_relative_hydra()
    args.logging.ckpt_dir = hydra_utils.to_absolute_path(args.logging.ckpt_dir)
    args.logging.tb_dir = hydra_utils.to_absolute_path(args.logging.tb_dir)
    args.data.train_filelist = hydra_utils.to_absolute_path(
        args.data.train_filelist)
    args.data.val_filelist = hydra_utils.to_absolute_path(
        args.data.val_filelist)
    for i in list(args.lincls.keys()):
        args.lincls[i].data.train_filelist = hydra_utils.to_absolute_path(
            args.lincls[i].data.train_filelist)
        args.lincls[i].data.val_filelist = hydra_utils.to_absolute_path(
            args.lincls[i].data.val_filelist)

    # If job is running, ignore
    jobnames = jobs_running()
    if args.logging.name.replace('.',
                                 '_').replace('-', '_') in jobnames and args.environment.slurm:
        print('Skipping {} because already in queue'.format(args.logging.name))
        return

    # If model is trained, ignore
    ckpt_fname = os.path.join(args.logging.ckpt_dir, args.logging.name,
                              'checkpoint_{:04d}.pth')
    if os.path.exists(ckpt_fname.format(args.optim.epochs)):
        all_exist = True
        model_name = ckpt_fname.format(args.optim.epochs)
        for dataset_i in list(args.lincls.keys()):
            lincls_fname = model_name + \
                args.lincls[dataset_i].eval_params.suffix + '.lincls'
            if not os.path.exists(lincls_fname):
                all_exist = False
        if all_exist:
            print('Skipping {}'.format(args.logging.name))
            return

    executor = submitit.AutoExecutor(
        folder=os.path.join(args.logging.submitit_dir,
                            '{}'.format(args.logging.name)),
        slurm_max_num_timeout=100,
        cluster=None if args.environment.slurm else "debug",
    )
    # asks SLURM to send USR1 signal 30 seconds before the time limit
    additional_parameters = {"signal": 'USR1@120'}
    if args.environment.nodelist != "":
        additional_parameters = {"nodelist": args.environment.nodelist}
    if args.environment.exclude_nodes != "":
        additional_parameters.update(
            {"exclude": args.environment.exclude_nodes})
    executor.update_parameters(
        timeout_min=args.environment.slurm_timeout,
        slurm_partition=args.environment.slurm_partition,
        cpus_per_task=args.environment.workers,
        gpus_per_node=args.environment.ngpu,
        nodes=args.environment.world_size,
        tasks_per_node=1,
        mem_gb=args.environment.mem_gb,
        slurm_additional_parameters=additional_parameters,
        signal_delay_s=120)
    executor.update_parameters(name=args.logging.name)
    job = executor.submit(Worker(), args)
    if not args.environment.slurm:
        job.result()


if __name__ == '__main__':
    main()
