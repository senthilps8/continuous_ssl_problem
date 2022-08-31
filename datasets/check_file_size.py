import numpy as np
import os
import tqdm
import multiprocessing as mp

MAX_LEN = 120
FOLDER = f'/grogu/user/pmorgado/datasets/flickrDB/'
CHUNK_SIZE = 50000
NUM_WORKERS = 40


def worker(start_index):
    mmap = np.memmap(f"{FOLDER}/filelist_200M.memmap",
                     dtype=f"S{MAX_LEN}",
                     mode="r",
                     shape=(int(200 * 1e6),))
    out_mmaps = {k: np.memmap(f"{FOLDER}/filelist_{k}M-v2.memmap",
                              dtype=f"S{MAX_LEN}",
                              mode="r+",
                              shape=(int(k * 1e6),))
                 for k in [1, 2, 5, 10, 20, 50, 100]}

    for index in range(start_index, start_index+CHUNK_SIZE):
        try:
            filename = mmap[index].decode('utf-8').strip()
            if os.path.getsize(filename) > 0:
                for DB_SZ in [1, 2, 5, 10, 20, 50, 100]:
                    if index < DB_SZ * 1e6:
                        out_mmaps[DB_SZ][index] = mmap[index]
            else:
                print(filename)
        except Exception:
            print(filename)
    for DB_SZ in [1, 2, 5, 10, 20, 50, 100]:
        out_mmaps[DB_SZ].flush()


if __name__ == '__main__':
    # pool = mp.Pool(NUM_WORKERS)
    # jobs = range(0, int(200*1e6), CHUNK_SIZE)
    # for _ in tqdm.tqdm(pool.imap_unordered(worker, jobs), total=len(jobs)):
    #     pass

    mmap = np.memmap(f"{FOLDER}/filelist_200M.memmap",
                     dtype=f"S{MAX_LEN}",
                     mode="r",
                     shape=(int(200 * 1e6),))
    out_mmaps = {k: np.memmap(f"{FOLDER}/filelist_{k}M-v2.memmap",
                              dtype=f"S{MAX_LEN}",
                              mode="r+",
                              shape=(int(k * 1e6),))
                 for k in [1, 2, 5, 10, 20, 50, 100]}
    for DB_SZ in [1, 2, 5, 10, 20, 50, 100]:
        extra_idx = int(100 * 1e6)
        for index in tqdm.tqdm(range(int(DB_SZ * 1e6)), total=int(DB_SZ * 1e6)):
            filename = out_mmaps[DB_SZ][index].decode('utf-8').strip()
            if len(filename) == 0:
                out_mmaps[DB_SZ][index] = mmap[extra_idx]
                extra_idx += 1
