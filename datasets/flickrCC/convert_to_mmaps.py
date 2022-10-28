import argparse
import numpy as np
import gzip


def convert2memmap(fn, size):
    memmap_fn = fn.replace('.tsv.gz', '.memmap')
    mmap = np.memmap(memmap_fn, dtype=f"S120", mode="w+", shape=(size,))

    with gzip.open(fn, 'rb') as f:
        for i, ln in enumerate(f):
            if i == len(mmap):
                break
            fn = ln.decode('utf-8').strip().split('\t')[-1]
            assert len(fn) < 120
            fn + '\n' + ' ' * (120 - 1 - len(fn))
            mmap[i] = fn
        mmap.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', default='lists/flickr-20M.tsv.gz', help='Subset file.')
    parser.add_argument('--subset_size', type=int, default=20000000, help='Subset file.')
    args = parser.parse_args()

    convert2memmap(args.subset, args.subset_size)


if __name__ == '__main__':
    main()
