import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as transF
import torch.utils.data as data
import numpy as np
import pdb
import itertools
import sys
import torch
import os
from scipy.stats import gamma
from collections import defaultdict


def encode_filename(fn, max_len=200):
    assert len(
        fn
    ) < max_len, f"Filename is too long. Specified max length is {max_len}"
    fn = fn + '\n' + ' ' * (max_len - len(fn))
    fn = np.fromstring(fn, dtype=np.uint8)
    fn = torch.ByteTensor(fn)
    return fn


def decode_filename(fn):
    fn = fn.cpu().numpy().astype(np.uint8)
    fn = fn.tostring().decode('utf-8')
    fn = fn.split('\n')[0]
    return fn


class KineticsSequentialDataset(data.Dataset):
    """Dataset that reads videos"""
    def __init__(self,
                 dirlist_fname,
                 base_dir='',
                 transforms=None,
                 fname_fmt='{:03d}.jpeg',
                 n_seq_samples=-1, sampling_method='consecutive'):
        """TODO: to be defined.

        :pair_filelist: TODO

        """
        data.Dataset.__init__(self)
        assert (os.path.exists(dirlist_fname)
                ), '{} does not exist'.format(dirlist_fname)
        with open(dirlist_fname, 'r') as f:
            filedata = f.read().splitlines()
            self.dirlist = torch.stack(
                [encode_filename(base_dir+'/'+d.split(' ')[0]) for d in filedata])
            self.nframes_list = torch.tensor(
                [int(d.split(' ')[1]) for d in filedata])
            print([decode_filename(fn) for fn in self.dirlist[:10]])

        self.num_videos = len(self.dirlist)
        self.sampling_method=sampling_method

        self.fname_fmt = fname_fmt
        self.n_seq_samples = n_seq_samples
        self.video_seq_starts = [0] * len(self.dirlist)
        if self.n_seq_samples > 0:
            self.vid_inds = np.cumsum(
                [0] + [self.n_seq_samples for i in range(self.num_videos)])
        else:
            self.vid_inds = np.cumsum(
                [0] + [self.nframes_list[i] for i in range(self.num_videos)])

        if not isinstance(transforms, list):
            transforms = [transforms]
        self.transforms = transforms
        self.std = torch.Tensor(self.transforms[0].transforms[-1].std).view(
            3, 1, 1)
        self.mean = torch.Tensor(self.transforms[0].transforms[-1].mean).view(
            3, 1, 1)

    def get_frame_filename(self, index):
        vid_index = np.searchsorted(self.vid_inds, index, side='right') - 1
        frame_start_index = 0
        if self.n_seq_samples > 0 and self.sampling_method=='consecutive':
            if index % self.n_seq_samples == 0:
                self.video_seq_starts[vid_index] = np.random.randint(
                    0,
                    max(
                        self.nframes_list[vid_index] -
                        self.n_seq_samples - 1, 1))
            frame_start_index = self.video_seq_starts[vid_index]

        if self.sampling_method=='consecutive':
            frame_index = (index -
                           self.vid_inds[vid_index]) + frame_start_index
            frame_index = min(frame_index,
                              self.nframes_list[vid_index] - 1)
        elif self.sampling_method=='random':
            frame_index = np.random.randint(0,self.nframes_list[vid_index])

        fname = os.path.join(decode_filename(self.dirlist[vid_index]),
                             self.fname_fmt.format(frame_index + 1))
        return fname

    def __getitem__(self, index):
        """TODO: Docstring for __getitem__.

        :index: TODO
        :returns: TODO

        """
        MAX_TRIES = 50
        for i in range(MAX_TRIES):
            try:
                fname = self.get_frame_filename(index)
                im1 = datasets.folder.pil_loader(fname)
                im2 = im1.copy()
                break

            except Exception:
                if i == MAX_TRIES - 1:
                    raise ValueError(
                        f'Aborting. Failed to load {MAX_TRIES} times in a row. Check {fname}'
                    )
                print('Failed to load')
                index = np.random.randint(len(self))
        meta = {}
        i = 0
        transform = None
        if self.transforms is not None:
            i = np.random.randint(len(self.transforms))
            transform = self.transforms[i]
        if transform is not None:
            im1 = transform(im1)
            im2 = transform(im2)
        meta['transind'] = i
        meta['fn'] = fname
        meta['fn1'] = fname
        meta['fn2'] = fname
        meta['index'] = index

        out = {
            'input1': im1,
            'input2': im2,
            'meta': meta,
        }
        return out

    def __len__(self):
        """TODO: Docstring for __len__.

        :f: TODO
        :returns: TODO

        """
        return self.vid_inds[-1]


class ClassSequentialDataset(data.Dataset):
    """Dataset that loads images"""
    def __init__(self,
                 train_filelist,
                 transforms=None,
                 fname_fmt='{:03d}.jpeg',
                 n_seq_samples=-1):
        """TODO: to be defined.

        :pair_filelist: TODO

        """
        data.Dataset.__init__(self)
        assert (os.path.exists(train_filelist)
                ), '{} does not exist'.format(train_filelist)

        with open(train_filelist, 'r') as f:
            all_files = f.read().splitlines()
            print('\n'.join([fn for fn in all_files[:10]]))

        all_labels = [fn.split('/')[-2] for fn in all_files]
        label_set = sorted(list(set(all_labels)))
        label_set = {y: i for i, y in enumerate(label_set)}
        all_labels = [label_set[lbl] for lbl in all_labels]

        # Define order
        filenames = all_files
        labels = all_labels
        if n_seq_samples > 0:
            from collections import defaultdict
            class2filelist = defaultdict(list)
            for fn, lbl in zip(all_files, all_labels):
                class2filelist[lbl] += [fn]

            # Shuffle images within each class
            import random
            for lbl in class2filelist:
                random.shuffle(class2filelist[lbl])

            # Sample a class, add n_seq_samples images to filelist
            filenames, labels = [], []
            while len(class2filelist) > 0:
                lbl = random.sample(class2filelist.keys(), 1)[0]
                files = class2filelist[lbl][:n_seq_samples]
                filenames += files
                labels += [lbl] * len(files)
                class2filelist[lbl] = class2filelist[lbl][n_seq_samples:]
                if len(class2filelist[lbl]) == 0:
                    del class2filelist[lbl]

        self.filenames = torch.stack([encode_filename(fn) for fn in filenames])
        self.labels = torch.tensor(labels)

        import torch.distributed as dist
        self.filenames = self.filenames.cuda()
        dist.broadcast(self.filenames, 0)
        self.filenames = self.filenames.cpu()

        self.labels = self.labels.cuda()
        dist.broadcast(self.labels, 0)
        self.labels = self.labels.cpu()

        print([decode_filename(fn) for fn in self.filenames[:100]])

        self.num_videos = self.filenames.shape[0]

        if not isinstance(transforms, list):
            transforms = [transforms]
        self.transforms = transforms

    def __getitem__(self, index):
        """TODO: Docstring for __getitem__.

        :index: TODO
        :returns: TODO

        """
        MAX_TRIES = 50
        for i in range(MAX_TRIES):
            try:
                fname = decode_filename(self.filenames[index])
                im1 = datasets.folder.pil_loader(fname)
                im2 = im1.copy()
                break

            except Exception:
                if i == MAX_TRIES - 1:
                    raise ValueError(
                        f'Aborting. Failed to load {MAX_TRIES} times in a row. Check {fname}'
                    )
                print(f'Failed to load. {fname}')
                index = np.random.randint(len(self))
        meta = {}
        i = 0
        transform = None
        if self.transforms is not None:
            i = np.random.randint(len(self.transforms))
            transform = self.transforms[i]
        if transform is not None:
            im1 = transform(im1)
            im2 = transform(im2)
        meta['transind'] = i
        meta['fn'] = fname
        meta['fn1'] = fname
        meta['fn2'] = fname
        meta['index'] = index
        meta['label'] = self.labels[index].item()

        out = {
            'input1': im1,
            'input2': im2,
            'meta': meta,
        }
        return out

    def __len__(self):
        """TODO: Docstring for __len__.

        :f: TODO
        :returns: TODO

        """
        return self.filenames.shape[0]


class ImageListDataset(data.Dataset):
    """Dataset that reads videos"""
    def __init__(self, list_fname, base_dir='', transforms=None):
        """TODO: to be defined.

        :pair_filelist: TODO

        """
        data.Dataset.__init__(self)
        assert (
            os.path.exists(list_fname)), '{} does not exist'.format(list_fname)
        with open(list_fname, 'r') as f:
            filedata = f.read().splitlines()
            self.filelist = torch.stack(
                [encode_filename(base_dir+"/"+d.split(' ')[0]) for d in filedata])
            print([decode_filename(fn) for fn in self.filelist[:10]])

        if not isinstance(transforms, list):
            transforms = [transforms]
        self.transforms = transforms
        self.std = torch.Tensor(self.transforms[0].transforms[-1].std).view(3, 1, 1)
        self.mean = torch.Tensor(self.transforms[0].transforms[-1].mean).view(3, 1, 1)

    def __getitem__(self, index):
        """TODO: Docstring for __getitem__.

        :index: TODO
        :returns: TODO

        """
        MAX_TRIES = 50
        for i in range(MAX_TRIES):
            try:
                fname = decode_filename(self.filelist[index])
                im1 = datasets.folder.pil_loader(fname)
                im2 = datasets.folder.pil_loader(fname)
                break

            except Exception:
                if i == MAX_TRIES - 1:
                    raise ValueError(
                        f'Aborting. Failed to load {MAX_TRIES} times in a row.'
                    )
                print('Failed to load')
                index = np.random.randint(len(self.filelist))
        meta = {}
        i = 0
        transform = None
        if self.transforms is not None:
            i = np.random.randint(len(self.transforms))
            transform = self.transforms[i]
        if transform is not None:
            im1 = transform(im1)
            im2 = transform(im2)
        meta['transind'] = i
        meta['fn'] = fname
        meta['fn1'] = fname
        meta['fn2'] = fname
        meta['index'] = index

        out = {
            'input1': im1,
            'input2': im2,
            'meta': meta,
        }
        return out

    def __len__(self):
        """TODO: Docstring for __len__.

        :f: TODO
        :returns: TODO

        """
        return len(self.filelist)


class MemMapImageListDataset(data.Dataset):
    """Dataset that reads videos"""
    def __init__(self,
                 list_fname,
                 num_files,
                 base_dir,
                 fn_dtype="S120",
                 subsample=1,
                 transforms=None):
        """TODO: to be defined.

        :pair_filelist: TODO

        """
        print(base_dir)
        data.Dataset.__init__(self)
        self.list_fname = list_fname
        self.num_files = num_files
        self.fn_dtype = fn_dtype
        self.base_dir = base_dir
        self.filelist_mmap = None
        self.subsample = subsample
        assert os.path.exists(list_fname), '{} does not exist'.format(list_fname)

        if not isinstance(transforms, list):
            transforms = [transforms]
        self.transforms = transforms
        self.std = torch.Tensor(self.transforms[0].transforms[-1].std).view(
            3, 1, 1)
        self.mean = torch.Tensor(self.transforms[0].transforms[-1].mean).view(
            3, 1, 1)

    def __getitem__(self, index):
        """TODO: Docstring for __getitem__.

        :index: TODO
        :returns: TODO

        """
        index = index * self.subsample
        if self.filelist_mmap is None:
            self.filelist_mmap = np.memmap(
                self.list_fname,
                dtype=self.fn_dtype,
                shape=(self.num_files, ),
                mode='r')

        MAX_TRIES = 50
        for i in range(MAX_TRIES):
            try:
                fname = self.filelist_mmap[index].decode('utf-8').strip().replace('.jpg', '-256p.jpg')
                fname = f"{self.base_dir}/{fname}"
                im = datasets.folder.pil_loader(fname)
                break
            except Exception:
                if i == MAX_TRIES - 1:
                    raise ValueError(
                        f'Aborting. Failed to load {MAX_TRIES} times in a row.'
                    )
                print(f'Failed to load: {fname}')
                index = np.random.randint(len(self))
                sys.stdout.flush()

        meta = {}
        i = 0
        transform = None
        if self.transforms is not None:
            i = np.random.randint(len(self.transforms))
            transform = self.transforms[i]
        if transform is not None:
            im1 = transform(im)
            im2 = transform(im)
        meta['transind'] = i
        meta['fn'] = fname
        meta['fn1'] = fname
        meta['fn2'] = fname
        meta['index'] = index

        out = {
            'input1': im1,
            'input2': im2,
            'meta': meta,
        }
        return out

    def __len__(self):
        """TODO: Docstring for __len__.

        :f: TODO
        :returns: TODO

        """
        return self.num_files // self.subsample


class ImageListStandardDataset(data.Dataset):
    """Dataset that reads videos"""
    def __init__(self, list_fname, transform=None):
        """TODO: to be defined.

        :pair_filelist: TODO

        """
        data.Dataset.__init__(self)
        assert (
            os.path.exists(list_fname)), '{} does not exist'.format(list_fname)
        if os.path.isdir(list_fname):
            import glob
            files = glob.glob(list_fname + '/*/*')
            classes = sorted(list(set([fn.split('/')[-2] for fn in files])))
            labels = [classes.index(fn.split('/')[-2]) for fn in files]
            self.pair_filelist = [(fn, lbl) for fn, lbl in zip(files, labels)]
        else:
            with open(list_fname, 'r') as f:
                filedata = f.read().splitlines()
                self.pair_filelist = [(d.split(' ')[0], int(d.split(' ')[1]))
                                      for d in filedata]

        self.transform = transform

    def __getitem__(self, index):
        """TODO: Docstring for __getitem__.

        :index: TODO
        :returns: TODO

        """
        fname1, target = self.pair_filelist[index]
        im1 = datasets.folder.pil_loader(fname1)
        if self.transform is not None:
            im1 = self.transform(im1)

        out = {
            'input': im1,
            'target': torch.tensor(target),
            'fname': fname1,
        }
        return out

    def __len__(self):
        """TODO: Docstring for __len__.

        :f: TODO
        :returns: TODO

        """
        return len(self.pair_filelist)
