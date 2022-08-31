import gzip
import os
import glob
import json
import sys
from PIL import Image

BASE = '/grogu/user/pmorgado/datasets/flickrDB'
# MAX_IMAGES = int(sys.argv[1])
MAX_IMAGES = 400*1e6
print(MAX_IMAGES)


# def get_image_id(row):
#     return row.decode('utf-8').split('\t')[2].split('/')[-1].split('.jpg')[0]


# def get_iids(fn):
#     meta = list(gzip.open(fn))
#     iids = set(get_image_id(m) for m in meta)
#     return iids


def get_image_id(filename):
    return filename.split('/')[-1].split('-256p.jpg')[0]


def get_iids(bucket):
    filenames = glob.glob(f"{BASE}/images/{bucket[0]}/{bucket[1]}/{bucket[2]}/{bucket}/*")
    good = []
    print(bucket)
    for fn in filenames:
        try:
            Image.open(fn)
            good += [fn]
        except Exception:
            print(fn)

    return set(get_image_id(fn) for fn in filenames)


def process_bucket(bk):
    bk = f"{bk:04d}"
    outp_fn = f'{BASE}/good_files/{bk}.txt'
    if os.path.exists(outp_fn):
        return
    image_iids = get_iids(bk)
    image_files = [f"{BASE}/images/{bk[0]}/{bk[1]}/{bk[2]}/{bk}/{iid}-256p.jpg"
                   for iid in image_iids]
    open(outp_fn, 'w').write('\n'.join(image_files))


def find_good_files(n_workers=10):
    import multiprocessing as mp
    import tqdm

    os.makedirs(f'{BASE}/good_files', exist_ok=True)
    pool = mp.Pool(n_workers)
    for _ in tqdm.tqdm(pool.imap_unordered(process_bucket, range(4500)), total=4500):
        pass

    image_files = []
    for bk in range(4500):
        if bk % 100 == 0:
            print(bk)
        bk = f"{bk:04d}"
        image_files += [ln.strip() for ln in open(f'{BASE}/good_files/{bk}.txt')]

    return image_files


def main():
    import numpy as np
    filenames = find_good_files(n_workers=250)
    MAX_LEN = 120

    for nfiles in [1, 2, 5, 10, 20, 50, 100, 200]:
        list_fn = f'{BASE}/filelist_{nfiles}M'
        print(list_fn)
        open(list_fn, 'w').write(
            "\n".join(filenames[:int(nfiles*1e6)]))

        mmap_fn = f"{BASE}/filelist_{nfiles}M.memmap"
        mmap = np.memmap(mmap_fn,
                         dtype=f"S{MAX_LEN}",
                         mode="w+",
                         shape=(int(nfiles*1e6),))
        for i, fn in enumerate(filenames[:int(nfiles*1e6)]):
            fn + '\n' + ' ' * (MAX_LEN - 1 - len(fn))
            mmap[i] = fn


if __name__ == '__main__':
    main()
